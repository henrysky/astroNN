# ---------------------------------------------------------#
# astroNN.NN.train_tools: Tools to train models
# ---------------------------------------------------------#

import numpy as np
import random
from astropy.io import  fits
import os
import astroNN.apogeetools.downloader

_APOGEE_DATA = os.getenv('SDSS_LOCAL_SAS_MIRROR')


def load_batch(num_train, batch_size, indx, mu_std, spectra, y):
    # Generate list of random indices (within the relevant partition of the main data file, e.g. the
    # training set) to be used to index into data_file
    indices = random.sample(range(indx, indx + num_train), batch_size)
    indices = np.sort(indices)

    mean_labels = mu_std[0]
    std_labels = mu_std[1]

    # load data
    spectra = spectra[indices, :]
    y = y[:][indices]

    # Normalize labels
    normed_y = (y - mean_labels) / std_labels

    # Reshape X data for compatibility with CNN
    spectra = spectra.reshape(len(spectra), spectra.shape[1], 1)

    return spectra, normed_y


def generate_train_batch(num_objects, batch_size, indx, mu_std, spectra, y):
    while True:
        x_batch, y_batch = load_batch(num_objects, batch_size, indx, mu_std, spectra, y)
        yield (x_batch, y_batch)


def generate_cv_batch(num_objects, batch_size, indx, mu_std, spectra, y):
    while True:
        x_batch, y_batch = load_batch(num_objects, batch_size, indx, mu_std, spectra, y)
        yield (x_batch, y_batch)


def apogee_id_fetch(relative_index=None, dr=None):
    """
    NAME: apogee_id_fetch
    PURPOSE: fetch apogee id from fits
    INPUT:
        relative_index in h5 file generated from h5_compiler
        dr = 13 or 14
    OUTPUT: real apogee_id
    HISTORY:
        2017-Oct-26 Henry Leung
    """
    if dr is None:
        dr = 14
        print('dr is not provided, using default dr=14')

    if dr == 14:
        allstarepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/allStar-l31c.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            astroNN.apogeetools.downloader.allstar(dr=14)
        hdulist = fits.open(allstarepath)
        apogee_id = hdulist[1].data['APOGEE_ID'][relative_index]
        if apogee_id.shape == (1,):
            apogee_id = str(apogee_id)
        else:
            apogee_id = np.array(apogee_id)
        return apogee_id
    else:
        raise ValueError('DR13 not supported')