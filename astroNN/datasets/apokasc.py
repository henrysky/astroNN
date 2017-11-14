# ---------------------------------------------------------#
#   astroNN.datasets.apokasc: compile h5 files for NN
# ---------------------------------------------------------#

from astroquery.vizier import Vizier
from astropy.io import fits
import numpy as np
import os
import pylab as plt

from astroNN.datasets.xmatch import xmatch
from astroNN.apogee.downloader import allstar, combined_spectra
from astroNN.apogee.apogee_shared import apogee_default_dr
from astroNN.shared.nn_tools import h5name_check
from astroNN.datasets.h5_compiler import gap_delete
from astroNN.apogee.downloader import allstarcannon

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from astropy.stats import mad_std


def apokasc_logg(dr=None, h5name=None, folder_name=None):

    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    h5name_check(h5name)
    dr = apogee_default_dr(dr=dr)

    catalog_list = Vizier.find_catalogs('apokasc')
    Vizier.ROW_LIMIT = 999999
    catalogs = Vizier.get_catalogs(catalog_list.keys())[1]
    apokasc_ra = catalogs['_RA']
    apokasc_dec = catalogs['_DE']
    apokasc_logg = catalogs['log_g_']

    allstarpath = allstar(dr=dr)
    hdulist = fits.open(allstarpath)
    print('Now processing allStar DR{} catalog'.format(dr))
    cannon_fullfilename = allstarcannon(dr=14)
    cannonhdulist = fits.open(cannon_fullfilename)

    apogee_ra = hdulist[1].data['RA']
    apogee_dec = hdulist[1].data['DEC']

    m1, m2, sep = xmatch(apogee_ra, apokasc_ra, maxdist=2, colRA1=apogee_ra, colDec1=apogee_dec, epoch1=2000.,
                         colRA2=apokasc_ra, colDec2=apokasc_dec, epoch2=2000., colpmRA2=None, colpmDec2=None, swap=True)
    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')

    apokasc_logg = apokasc_logg[m2]
    aspcap_residue = []
    astronn_residue = []
    cannon_residue = []
    i = 0
    std_labels = np.std(apokasc_logg)

    for index in m1:
        apogee_id = hdulist[1].data['APOGEE_ID'][index]
        location_id = hdulist[1].data['LOCATION_ID'][index]
        cannon_residue.extend([cannonhdulist[1].data['LOGG'][index] - apokasc_logg[i]])
        warningflag, path = combined_spectra(dr=dr, location=location_id, apogee=apogee_id, verbose=0)
        if warningflag is None:
            combined_file = fits.open(path)
            _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
            spec = (_spec - spec_meanstd[0])/spec_meanstd[1]
            spec = gap_delete(spec, dr=14)
            aspcap_residue.extend([hdulist[1].data['PARAM'][index, 1] - apokasc_logg[i]])
            prediction = model.predict(spec.reshape([1, len(spec), 1]), batch_size=1)
            prediction *= mean_and_std[1]
            prediction += mean_and_std[0]
            astronn_residue.extend([prediction[0,1] - apokasc_logg[i]])
        i += 1


    plt.figure(figsize=(15, 11), dpi=200)
    plt.axhline(0, ls='--', c='k', lw=2)
    plt.scatter(apokasc_logg, aspcap_residue, s=3)
    fullname = 'Log(g)'
    x_lab = 'APOKASC'
    y_lab = 'ASPCAP'
    plt.xlabel('APOKASC ' + fullname, fontsize=25)
    plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.xlim([np.min(apokasc_logg), np.max(apokasc_logg)])
    ranges = (np.max(apokasc_logg) - np.min(apokasc_logg)) / 2
    plt.ylim([-ranges, ranges])
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
    plt.figtext(0.6, 0.75,
                '$\widetilde{m}$=' + '{0:.3f}'.format(np.median(aspcap_residue)) + ' s=' +
                '{0:.3f}'.format(mad_std(aspcap_residue)), size=25, bbox=bbox_props)
    plt.tight_layout()
    plt.savefig(fullfolderpath + '/apokasc_aspcap_logg.png')
    plt.close('all')
    plt.clf()

    plt.figure(figsize=(15, 11), dpi=200)
    plt.axhline(0, ls='--', c='k', lw=2)
    plt.scatter(apokasc_logg, astronn_residue, s=3)
    fullname = 'Log(g)'
    x_lab = 'APOKASC'
    y_lab = 'astroNN'
    plt.xlabel('APOKASC ' + fullname, fontsize=25)
    plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.xlim([np.min(apokasc_logg), np.max(apokasc_logg)])
    ranges = (np.max(apokasc_logg) - np.min(apokasc_logg)) / 2
    plt.ylim([-ranges, ranges])
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
    plt.figtext(0.6, 0.75,
                '$\widetilde{m}$=' + '{0:.3f}'.format(np.median(astronn_residue)) + ' s=' +
                '{0:.3f}'.format(mad_std(astronn_residue)), size=25, bbox=bbox_props)
    plt.tight_layout()
    plt.savefig(fullfolderpath + '/apokasc_astroNN_logg.png')
    plt.close('all')
    plt.clf()

    plt.figure(figsize=(15, 11), dpi=200)
    plt.axhline(0, ls='--', c='k', lw=2)
    plt.scatter(apokasc_logg, cannon_residue, s=3)
    fullname = 'Log(g)'
    x_lab = 'APOKASC'
    y_lab = 'Cannon'
    plt.xlabel('APOKASC ' + fullname, fontsize=25)
    plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.xlim([np.min(apokasc_logg), np.max(apokasc_logg)])
    ranges = (np.max(apokasc_logg) - np.min(apokasc_logg)) / 2
    plt.ylim([-ranges, ranges])
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
    plt.figtext(0.6, 0.75,
                '$\widetilde{m}$=' + '{0:.3f}'.format(np.median(cannon_residue)) + ' s=' +
                '{0:.3f}'.format(mad_std(cannon_residue)), size=25, bbox=bbox_props)
    plt.tight_layout()
    plt.savefig(fullfolderpath + '/apokasc_cannon_logg.png')
    plt.close('all')
    plt.clf()

