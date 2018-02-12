# ---------------------------------------------------------#
#   astroNN.datasets.apogee_rc: APOGEE RC
# ---------------------------------------------------------#

import os

import numpy as np
import pylab as plt
from astroNN.apogee.apogee_shared import apogee_default_dr
from astroNN.apogee.chips import gap_delete
from astroNN.apogee.downloader import allstar, combined_spectra
from astroNN.apogee.downloader import apogee_vac_rc
from astroNN.gaia.gaia_shared import mag_to_absmag, mag_to_fakemag
from astroNN.shared.nn_tools import gpu_memory_manage
from astropy import units as u
from astropy.io import fits
from astropy.stats import mad_std
from keras.models import load_model


def load_apogee_rc(dr=None, metric='distance'):
    """
    NAME:
        load_apogee_rc
    PURPOSE:
        load apogee red clumps (absolute magnitude measurement)
    INPUT:
    OUTPUT:
    HISTORY:
        2018-Jan-21 - Written - Henry Leung (University of Toronto)
    """
    fullfilename = apogee_vac_rc(dr=dr, verbose=1)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        RA = hdulist['RA']
        DEC = hdulist['DEC']
        rc_dist = hdulist['RC_DIST']
        rc_parallax = (1 / rc_dist) * u.mas  # Convert kpc to parallax in mas
        K_mag = hdulist['K']

    if metric == 'distance':
        output = rc_dist * 1000 * u.parsec

    elif metric == 'absmag':
        absmag = mag_to_absmag(K_mag, 1 / rc_dist * u.arcsec)
        output = absmag

    elif metric == 'fakemag':
        # fakemag requires parallax (mas)
        fakemag = mag_to_fakemag(K_mag, 1000 / rc_dist * u.mas)
        output = fakemag

    else:
        raise ValueError('Unknown metric')

    return RA, DEC, output


def apogee_rc(dr=None, folder_name=None):
    """
    NAME: apogee_rc_absmag
    PURPOSE: check red clumps
    INPUT:
        dr = 14
        h5name = name of h5 dataset you want to create
    OUTPUT: plots
    HISTORY:
        2017-Nov-16 Henry Leung
    """

    # prevent Tensorflow taking up all the GPU memory
    gpu_memory_manage()

    dr = apogee_default_dr(dr=dr)

    warning_flag, fullfilename = apogee_vac_rc(dr=dr, verbose=1)

    with fits.open(fullfilename) as F:
        hdulist = F[1].data
        apogeee_id = hdulist['APOGEE_ID']
        location_id = hdulist['LOCATION_ID']
        rc_dist = hdulist['RC_DIST']
        rc_parallax = 1 / (rc_dist * 1000)
        k_mag_apogee = hdulist['K']
    absmag = mag_to_absmag(k_mag_apogee, rc_parallax)

    allstarpath = allstar(dr=dr)
    hdulist = fits.open(allstarpath)
    print('Now processing allStar DR{} catalog'.format(dr))

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]
    num_labels = mean_and_std.shape[1]

    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')

    astronn_absmag_resid = []

    spec = []
    absmag = absmag[:1000]

    for counter, id in enumerate(apogeee_id[:1000]):
        warningflag, path = combined_spectra(dr=dr, location=location_id[counter], apogee=id, verbose=0)
        combined_file = fits.open(path)
        _spec = combined_file[1].data
        _spec = gap_delete(_spec, dr=14)
        _spec = (_spec - spec_meanstd[0]) / spec_meanstd[1]
        spec.extend([_spec])
    spec = np.array(spec)
    prediction, model_uncertainty = batch_dropout_predictions(model, spec, 500, num_labels, std_labels, mean_labels)
    astronn_absmag_resid = prediction[:, 22] - absmag
    model_uncertainty = np.array(model_uncertainty[:, 22])

    hdulist.close()

    plt.figure(figsize=(15, 11), dpi=200)
    plt.axhline(0, ls='--', c='k', lw=2)
    plt.errorbar(absmag, astronn_absmag_resid, yerr=model_uncertainty, markersize=2,
                 fmt='o', ecolor='g', color='blue', capthick=2, elinewidth=0.5)
    fullname = 'Absolute Magnitude'
    x_lab = 'APOGEE Red Clumps'
    y_lab = 'astroNN'
    plt.xlabel('APOGEE Red Clumps ' + fullname, fontsize=25)
    plt.ylabel('$\Delta$ ' + fullname + '\n(' + y_lab + ' - ' + x_lab + ')', fontsize=25)
    plt.tick_params(labelsize=20, width=1, length=10)
    plt.xlim([np.min(absmag), np.max(absmag)])
    ranges = (np.max(absmag) - np.min(absmag)) / 2
    plt.ylim([-ranges, ranges])
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2)
    plt.figtext(0.6, 0.75,
                '$\widetilde{m}$=' + '{0:.3f}'.format(np.median(astronn_absmag_resid)) + ' s=' +
                '{0:.3f}'.format(mad_std(astronn_absmag_resid)), size=25, bbox=bbox_props)
    plt.tight_layout()
    plt.savefig(fullfolderpath + '/absmag_RC_astroNN.png')
    plt.close('all')
    plt.clf()
