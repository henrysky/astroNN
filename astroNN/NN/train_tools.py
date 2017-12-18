# ---------------------------------------------------------#
# astroNN.NN.train_tools: Tools to train models
# ---------------------------------------------------------#

import os
import threading

import astroNN.apogee.downloader
import keras.backend as K
from keras.callbacks import Callback
import numpy as np
from astropy.io import fits

_APOGEE_DATA = os.getenv('SDSS_LOCAL_SAS_MIRROR')


class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class DataGenerator(object):
    """
    NAME: DataGenerator
    PURPOSE: to generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 Henry Leung
    """

    def __init__(self, dim, batch_size, num_train, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_train = num_train

    @threadsafe_generator
    def generate(self, spectra, labels):
        'Generates batches of samples'
        # Infinite loop
        list_IDs = range(spectra.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                X, y = self.__data_generation(spectra, labels, list_IDs_temp)

                yield X, {'linear_output': y, 'lin_var_ouput': y}

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, spectra, labels, list_IDs_temp):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size, labels.shape[1]))

        # Generate data
        X[:, :, 0] = spectra[list_IDs_temp]
        y[:] = labels[list_IDs_temp]

        return X, y


def mean_squared_error(y_true, y_pred):
    """
    NAME: mean_squared_error
    PURPOSE: Custom loss function to do mean squared error loss
    INPUT:
        you should not use this function directly
    OUTPUT:
    HISTORY:
        2017-Dec-04 Henry Leung
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mse_var_wrap(linear_output):
    """
    NAME: mse_var_wrap
    PURPOSE: Just a function to wrap arounf mase_var for Keras but still getting num_labels variable from outside
    INPUT:
        you should not use this function directly
    OUTPUT:
    HISTORY:
    """
    def mse_var(y_true, y_pred):
        """
        NAME: mse_var
        PURPOSE: Custom loss function to do mean squared error loss with uncertainty into account
        INPUT:
            you should not use this function directly
        OUTPUT:
        HISTORY:
        """
        return K.mean(0.5*K.square(linear_output - y_true)*K.exp(-y_pred) + 0.5*y_pred,axis=-1)

    return mse_var


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1


def apogee_id_fetch(relative_index=None, dr=None):
    """
    NAME: apogee_id_fetch
    PURPOSE: fetch apogee id from fits
    INPUT:
        relative_index in h5 file generated from h5_compiler
        relative_index in h5 file generated from h5_compiler
        dr = 13 or 14
    OUTPUT: real apogee_id
    HISTORY:
        2017-Nov-03 Henry Leung
    """
    if dr is None:
        dr = 14
        print('dr is not provided, using default dr=14')
    if dr == 13:
        allstarepath = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/allStar-l30e.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            astroNN.apogee.downloader.allstar(dr=13)
        hdulist = fits.open(allstarepath)
        apogee_id = hdulist[1].data['APOGEE_ID'][relative_index]
        apogee_id = np.array(apogee_id)
        return apogee_id
    if dr == 14:
        allstarepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/allStar-l31c.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            astroNN.apogee.downloader.allstar(dr=14)
        hdulist = fits.open(allstarepath)
        apogee_id = hdulist[1].data['APOGEE_ID'][relative_index]
        apogee_id = np.array(apogee_id)
        return apogee_id
    else:
        raise ValueError('Only DR13/DR14 supported')