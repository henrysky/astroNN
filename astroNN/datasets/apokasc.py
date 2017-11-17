# ---------------------------------------------------------#
#   astroNN.datasets.apokasc: apokasc Log(g)
# ---------------------------------------------------------#

from astroquery.vizier import Vizier
from astropy.io import fits
import numpy as np
import os
import pylab as plt
import seaborn as sns

from astroNN.datasets.xmatch import xmatch
from astroNN.apogee.downloader import allstar, combined_spectra
from astroNN.apogee.apogee_shared import apogee_default_dr
from astroNN.datasets.h5_compiler import gap_delete
from astroNN.apogee.downloader import allstarcannon

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from astropy.stats import mad_std

import h5py


def apokasc_logg(dr=None, folder_name=None):
    """
    NAME: apokasc_logg
    PURPOSE: check apokasc result
    INPUT:
        dr = 14
        folder_name = the folder name contains the model
    OUTPUT: plots
    HISTORY:
        2017-Nov-15 Henry Leung
    """
    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    dr = apogee_default_dr(dr=dr)

    catalog_list = Vizier.find_catalogs('apokasc')
    Vizier.ROW_LIMIT = 5000
    catalogs_gold = Vizier.get_catalogs(catalog_list.keys())[1]
    catalogs_basic = Vizier.get_catalogs(catalog_list.keys())[2]
    apokasc_gold_ra = catalogs_gold['_RA']
    apokasc_gold_dec = catalogs_gold['_DE']
    apokasc_gold_logg = catalogs_gold['log_g_']
    apokasc_basic_ra = catalogs_basic['_RA']
    apokasc_basic_dec = catalogs_basic['_DE']
    apokasc_basic_logg = catalogs_basic['log.g2']

    allstarpath = allstar(dr=dr)
    hdulist = fits.open(allstarpath)
    print('Now processing allStar DR{} catalog'.format(dr))
    cannon_fullfilename = allstarcannon(dr=14)
    cannonhdulist = fits.open(cannon_fullfilename)

    apogee_ra = hdulist[1].data['RA']
    apogee_dec = hdulist[1].data['DEC']

    m1_basic, m2_basic, sep = xmatch(apogee_ra, apokasc_basic_ra, maxdist=2, colRA1=apogee_ra, colDec1=apogee_dec, epoch1=2000.,
                         colRA2=apokasc_basic_ra, colDec2=apokasc_basic_dec, epoch2=2000., colpmRA2=None, colpmDec2=None, swap=True)

    m1_gold, m2_gold, sep = xmatch(apogee_ra, apokasc_gold_ra, maxdist=2, colRA1=apogee_ra, colDec1=apogee_dec, epoch1=2000.,
                         colRA2=apokasc_gold_ra, colDec2=apokasc_gold_dec, epoch2=2000., colpmRA2=None, colpmDec2=None, swap=True)

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')

    apokasc_basic_logg = np.array(apokasc_basic_logg[m2_basic])
    useless = np.argwhere(np.isnan(apokasc_basic_logg))
    m1_basic = np.delete(m1_basic, useless)
    apokasc_basic_logg = np.delete(apokasc_basic_logg, useless)
    apokasc_gold_logg = apokasc_gold_logg[m2_gold]

    aspcap_basic_residue = []
    astronn_basic_residue = []
    cannon_basic_residue = []

    aspcap_gold_residue = []
    astronn_gold_residue = []
    cannon_gold_residue = []

    for counter, index in enumerate(m1_basic):
        apogee_id = hdulist[1].data['APOGEE_ID'][index]
        location_id = hdulist[1].data['LOCATION_ID'][index]
        cannon_basic_residue.extend([cannonhdulist[1].data['LOGG'][index] - apokasc_basic_logg[counter]])
        warningflag, path = combined_spectra(dr=dr, location=location_id, apogee=apogee_id, verbose=0)
        if warningflag is None:
            combined_file = fits.open(path)
            _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
            spec = (_spec - spec_meanstd[0])/spec_meanstd[1]
            spec = gap_delete(spec, dr=14)
            aspcap_basic_residue.extend([hdulist[1].data['PARAM'][index, 1] - apokasc_basic_logg[counter]])
            prediction = model.predict(spec.reshape([1, len(spec), 1]), batch_size=1)
            prediction *= mean_and_std[1]
            prediction += mean_and_std[0]
            astronn_basic_residue.extend([prediction[0,1] - apokasc_basic_logg[counter]])

    for counter, index in enumerate(m1_gold):
        apogee_id = hdulist[1].data['APOGEE_ID'][index]
        location_id = hdulist[1].data['LOCATION_ID'][index]
        cannon_gold_residue.extend([cannonhdulist[1].data['LOGG'][index] - apokasc_gold_logg[counter]])
        warningflag, path = combined_spectra(dr=dr, location=location_id, apogee=apogee_id, verbose=0)
        if warningflag is None:
            combined_file = fits.open(path)
            _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
            spec = gap_delete(_spec, dr=14)
            spec = (spec - spec_meanstd[0])/spec_meanstd[1]
            aspcap_gold_residue.extend([hdulist[1].data['PARAM'][index, 1] - apokasc_gold_logg[counter]])
            prediction = model.predict(spec.reshape([1, len(spec), 1]), batch_size=1)
            prediction *= mean_and_std[1]
            prediction += mean_and_std[0]
            astronn_gold_residue.extend([prediction[0,1] - apokasc_gold_logg[counter]])

    hdulist.close()
    cannonhdulist.close()

    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    for i in ['ASPCAP', 'astroNN', 'Cannon']:
        if i=='ASPCAP':
            resid_basic = np.array(aspcap_basic_residue)
            resid_gold = aspcap_gold_residue
        elif i=='astroNN':
            resid_basic = astronn_basic_residue
            resid_gold = astronn_gold_residue
        elif i=='Cannon':
            resid_basic = cannon_basic_residue
            resid_gold = cannon_gold_residue
        plt.figure(figsize=(15, 11), dpi=200)
        plt.axhline(0, ls='--', c='k', lw=2)
        plt.scatter(apokasc_basic_logg, resid_basic, s=3, label = 'APOKASC Basic')
        plt.scatter(apokasc_gold_logg, resid_gold, s=3, label = 'APOKASC Gold Standard')
        fullname = 'Log(g)'
        x_lab = 'APOKASC'
        y_lab = str(i)
        plt.xlabel('APOKASC ' + fullname, fontsize=25)
        plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
        plt.tick_params(labelsize=20, width=1, length=10)
        plt.xlim([np.min(apokasc_basic_logg), np.max(apokasc_basic_logg)])
        ranges = (np.max(apokasc_basic_logg) - np.min(apokasc_basic_logg)) / 2
        plt.ylim([-ranges, ranges])
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
        plt.figtext(0.5, 0.8,
                    'Basic: $\widetilde{m}$=' + '{0:.3f}'.format(np.median(resid_basic)) + ' s=' +
                    '{0:.3f}'.format(mad_std(resid_basic)), size=25, bbox=bbox_props)
        plt.figtext(0.2, 0.8,
                    'Gold: $\widetilde{m}$=' + '{0:.3f}'.format(np.median(resid_gold)) + ' s=' +
                    '{0:.3f}'.format(mad_std(resid_gold)), size=25, bbox=bbox_props)
        plt.legend(loc='best', fontsize=20, markerscale=6)
        plt.tight_layout()
        plt.savefig(fullfolderpath + '/apokasc_{}_logg.png'.format(i))
        plt.close('all')
        plt.clf()

    return None
