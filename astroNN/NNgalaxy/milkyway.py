# ---------------------------------------------------------#
#   astroNN.NNgalaxy.milkyway: plot milkyway from NN
# ---------------------------------------------------------#

from astropy.io import fits
import numpy as np
import os
import pylab as plt
from astropy.stats import mad_std

from astroNN.apogee.downloader import allstar, combined_spectra
from astroNN.apogee.apogee_shared import apogee_default_dr
from astroNN.apogee.downloader import apogee_vac_rc
from astroNN.datasets.h5_compiler import gap_delete
from astroNN.apogee.downloader import allstarcannon
from astroNN.gaia.gaia_shared import gaia_env, gaia_default_dr, mag_to_absmag

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

import h5py


def mw_maps(dr=None, folder_name=None):
    """

    ra= 120.*u.deg
    dec= 30.*u.deg
    distance= 1.2*u.kpc
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,frame='icrs')
    gc = c.transform_to(apycoords.Galactocentric)
    print("(x,y,z) in (kpc,kpc,kpc) in right-handed frame")
    print("\t",gc.cartesian)

    NAME: mw_maps
    PURPOSE: plot map of milkyway
    INPUT:
        dr = 14
        h5name = name of h5 dataset you want to create
    OUTPUT: plots
    HISTORY:
        2017-Nov-17 Henry Leung
    """
    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    dr = apogee_default_dr(dr=dr)

    warning_flag, fullfilename = apogee_vac_rc(dr=dr, verbose=1)

    hdulist = fits.open(fullfilename)
    apogeee_id = hdulist[1].data['APOGEE_ID']
    location_id = hdulist[1].data['LOCATION_ID']
    rc_dist = hdulist[1].data['RC_DIST']
    rc_parallax = 1 / (rc_dist * 1000)
    k_mag_apogee = hdulist[1].data['K']
    absmag = mag_to_absmag(k_mag_apogee, rc_parallax)

    allstarpath = allstar(dr=dr)
    hdulist = fits.open(allstarpath)
    print('Now processing allStar DR{} catalog'.format(dr))
    cannon_fullfilename = allstarcannon(dr=14)
    cannonhdulist = fits.open(cannon_fullfilename)

    currentdir = os.getcwd()
    fullfolderpath = currentdir + '/' + folder_name
    modelname = '/model_{}.h5'.format(folder_name[-11:])
    model = load_model(os.path.normpath(fullfolderpath + modelname))
    mean_and_std = np.load(fullfolderpath + '/meanstd.npy')
    spec_meanstd = np.load(fullfolderpath + '/spectra_meanstd.npy')

    astronn_absmag_resid = []

    for counter, id in enumerate(apogeee_id):
        warningflag, path = combined_spectra(dr=dr, location=location_id[counter], apogee=id, verbose=0)
        combined_file = fits.open(path)
        spec = combined_file[1].data
        spec = gap_delete(spec, dr=14)
        spec = (spec - spec_meanstd[0]) / spec_meanstd[1]
        prediction = model.predict(spec.reshape([1, len(spec), 1]), batch_size=1)
        prediction *= mean_and_std[1]
        prediction += mean_and_std[0]
        astronn_absmag_resid.extend([prediction[0, 22] - absmag[counter]])

    hdulist.close()
    cannonhdulist.close()

    plt.figure(figsize=(15, 11), dpi=200)
    plt.axhline(0, ls='--', c='k', lw=2)
    plt.scatter(absmag, astronn_absmag_resid, s=3)
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
