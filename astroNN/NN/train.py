# ---------------------------------------------------------#
#   astroNN.NN.train: train models
# ---------------------------------------------------------#

import h5py
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import astroNN.NN.cnn_models


def apogee_train(h5data=None, target=None, h5test=None):
    """
    NAME: load_batch_from_h5
    PURPOSE: To load batch for training from .h5 dataset file
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    if h5data is None:
        raise ValueError('Please specift the dataset name using h5data="......"')
    if target is None:
        raise ValueError('Please specift a list of target names using target=[.., ...]')
    if h5test is None:
        raise ValueError('Please specift the testset name using h5test="......"')

    num_labels = target.shape

    with h5py.File(h5data) as F:
        spectra = F['spectra']
        num_flux = spectra.shape[1]
        num_train = int(0.9*spectra.shape[0]) # number of training example, rest are cross validation
        num_cv = spectra.shape[0] - num_train # cross validation
    print('Each spectrum contains ' + str(num_flux) + ' wavelength bins')
    print('Training set includes ' + str(num_train) + ' spectra and the cross-validation set includes ' + str(num_cv)
          + ' spectra')

    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    return None