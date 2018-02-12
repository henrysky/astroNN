# ---------------------------------------------------------#
#   astroNN.datasets.apokasc: apokasc Log(g)
# ---------------------------------------------------------#

import os

import numpy as np
import pylab as plt
import seaborn as sns
from astroNN.apogee.apogee_shared import apogee_default_dr
from astroNN.apogee.chips import gap_delete
from astroNN.apogee.downloader import allstar, combined_spectra
from astroNN.apogee.downloader import allstarcannon
from astroNN.datasets.xmatch import xmatch
from astroNN.shared.nn_tools import gpu_memory_manage
from astropy.io import fits
from astropy.stats import mad_std
from astroquery.vizier import Vizier
from keras.models import load_model


def apokasc_load():
    """
    NAME:
        apokasc_load
    PURPOSE:
        load apokasc result (Precise surface gravity measurement)
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-23 - Written - Henry Leung (University of Toronto)
    """
    catalog_list = Vizier.find_catalogs('apokasc')
    Vizier.ROW_LIMIT = 99999
    catalogs_gold = Vizier.get_catalogs(catalog_list.keys())[1]
    catalogs_basic = Vizier.get_catalogs(catalog_list.keys())[2]
    gold_ra = catalogs_gold['_RA']
    gold_dec = catalogs_gold['_DE']
    gold_logg = catalogs_gold['log_g_']
    basic_ra = catalogs_basic['_RA']
    basic_dec = catalogs_basic['_DE']
    basic_logg = catalogs_basic['log.g2']

    return gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg


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
    gpu_memory_manage()

    dr = apogee_default_dr(dr=dr)

    apokasc_gold_ra, apokasc_gold_dec, apokasc_gold_logg, apokasc_basic_ra, apokasc_basic_dec, apokasc_basic_logg \
        = apokasc_load()

    allstarpath = allstar(dr=dr)
    hdulist = fits.open(allstarpath)
    print('Now processing allStar DR{} catalog'.format(dr))
    cannon_fullfilename = allstarcannon(dr=14)
    cannonhdulist = fits.open(cannon_fullfilename)

    apogee_ra = hdulist[1].data['RA']
    apogee_dec = hdulist[1].data['DEC']

    m1_basic, m2_basic, sep = xmatch(apogee_ra, apokasc_basic_ra, maxdist=2, colRA1=apogee_ra, colDec1=apogee_dec,
                                     epoch1=2000.,
                                     colRA2=apokasc_basic_ra, colDec2=apokasc_basic_dec, epoch2=2000., colpmRA2=None,
                                     colpmDec2=None, swap=True)

    m1_gold, m2_gold, sep = xmatch(apogee_ra, apokasc_gold_ra, maxdist=2, colRA1=apogee_ra, colDec1=apogee_dec,
                                   epoch1=2000.,
                                   colRA2=apokasc_gold_ra, colDec2=apokasc_gold_dec, epoch2=2000., colpmRA2=None,
                                   colpmDec2=None, swap=True)

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')

    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]
    num_labels = mean_and_std.shape[1]

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

    cannon_basic_uncertainty = []
    cannon_gold_uncertainty = []

    aspcap_basic_uncertainty = []
    aspcap_gold_uncertainty = []

    spec = []

    for counter, index in enumerate(m1_basic):
        apogee_id = hdulist[1].data['APOGEE_ID'][index]
        location_id = hdulist[1].data['LOCATION_ID'][index]
        cannon_basic_residue.extend([cannonhdulist[1].data['LOGG'][index] - apokasc_basic_logg[counter]])
        cannon_basic_uncertainty.extend([cannonhdulist[1].data['LOGG_ERR'][index]])
        aspcap_basic_uncertainty.extend([hdulist[1].data['LOGG_ERR'][index]])
        warningflag, path = combined_spectra(dr=dr, location=location_id, apogee=apogee_id, verbose=0)
        if warningflag is None:
            combined_file = fits.open(path)
            _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
            _spec = gap_delete(_spec, dr=14)
            _spec = (_spec - spec_meanstd[0]) / spec_meanstd[1]
            spec.extend([_spec])
            aspcap_basic_residue.extend([hdulist[1].data['PARAM'][index, 1] - apokasc_basic_logg[counter]])
    spec = np.array(spec)
    spec = spec.reshape(spec.shape[0], spec.shape[1], 1)
    prediction, model_uncertainty = batch_dropout_predictions(model, spec, 500, num_labels, std_labels, mean_labels)
    astronn_basic_residue = prediction[:, 1] - apokasc_basic_logg
    astronn_basic_uncertainty = model_uncertainty[:, 1]

    spec = []
    for counter, index in enumerate(m1_gold):
        apogee_id = hdulist[1].data['APOGEE_ID'][index]
        location_id = hdulist[1].data['LOCATION_ID'][index]
        cannon_gold_residue.extend([cannonhdulist[1].data['LOGG'][index] - apokasc_gold_logg[counter]])
        cannon_gold_uncertainty.extend([cannonhdulist[1].data['LOGG_ERR'][index]])
        aspcap_gold_uncertainty.extend([hdulist[1].data['LOGG_ERR'][index]])
        warningflag, path = combined_spectra(dr=dr, location=location_id, apogee=apogee_id, verbose=0)
        if warningflag is None:
            combined_file = fits.open(path)
            _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
            _spec = gap_delete(_spec, dr=14)
            _spec = (_spec - spec_meanstd[0]) / spec_meanstd[1]
            spec.extend([_spec])
            aspcap_gold_residue.extend([hdulist[1].data['PARAM'][index, 1] - apokasc_gold_logg[counter]])
            num_labels = mean_and_std.shape[1]
    spec = np.array(spec)
    prediction, model_uncertainty = batch_dropout_predictions(model, spec, 500, num_labels, std_labels, mean_labels)
    astronn_gold_residue = prediction[:, 1] - apokasc_gold_logg
    astronn_gold_uncertainty = model_uncertainty[:, 1]

    hdulist.close()
    cannonhdulist.close()

    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = '0.4'

    for i in ['ASPCAP', 'astroNN', 'Cannon']:
        if i == 'ASPCAP':
            resid_basic = np.array(aspcap_basic_residue)
            resid_gold = aspcap_gold_residue
            uncertainty_basic = np.array(aspcap_basic_uncertainty)
            uncertainty_gold = np.array(aspcap_gold_uncertainty)

        elif i == 'Cannon':
            resid_basic = cannon_basic_residue
            resid_gold = cannon_gold_residue
            uncertainty_basic = np.array(cannon_basic_uncertainty)
            uncertainty_gold = np.array(cannon_gold_uncertainty)
        elif i == 'astroNN':
            resid_basic = np.array(astronn_basic_residue)
            resid_gold = np.array(astronn_gold_residue)
            uncertainty_basic = np.array(astronn_basic_uncertainty)
            uncertainty_gold = np.array(astronn_gold_uncertainty)
        plt.figure(figsize=(15, 11), dpi=200)
        plt.axhline(0, ls='--', c='k', lw=2)
        plt.errorbar(apokasc_basic_logg, resid_basic, yerr=uncertainty_basic, markersize=2,
                     fmt='o', ecolor='g', color='blue', capthick=2, elinewidth=0.5, label='APOKASC Basic')
        plt.errorbar(apokasc_gold_logg, resid_gold, yerr=uncertainty_gold, markersize=2,
                     fmt='o', ecolor='g', color='orange', capthick=2, elinewidth=0.5, label='APOKASC Gold Standard')
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
