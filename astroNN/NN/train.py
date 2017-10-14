# ---------------------------------------------------------#
#   astroNN.NN.train: train models
# ---------------------------------------------------------#

import tensorflow as tf
import h5py
import random
import numpy as np


def load_batch_from_h5(data_file, num_objects, batch_size, indx, mu_std=''):
    """
    NAME: load_batch_from_h5
    PURPOSE: To load batch for training from .h5 dataset file
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    mean_and_std = np.load(mu_std)
    mean_labels = mean_and_std[0]
    std_labels = mean_and_std[1]

    # Generate list of random indices (within the relevant partition of the main data file, e.g. the
    # training set) to be used to index into data_file
    indices = random.sample(range(indx, indx + num_objects), batch_size)
    indices = np.sort(indices)

    # load data
    F = h5py.File(data_file, 'r')
    X = F['Spectra']
    teff = F['temp']
    logg = F['logg']
    fe_h = F['iron']

    X = X[indices, :]

    y = np.column_stack((teff[:][indices],
                         logg[:][indices],
                         fe_h[:][indices]))

    # Normalize labels
    normed_y = (y - mean_labels) / std_labels

    # Reshape X data for compatibility with CNN
    X = X.reshape(len(X), 7514, 1)

    return X, normed_y


def generate_train_batch(data_file, num_objects, batch_size, indx, mu_std):
    """
    NAME: generate_train_batch
    PURPOSE: To generate training batch
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    while True:
        x_batch, y_batch = load_batch_from_h5(data_file, num_objects, batch_size, indx, mu_std)
        yield (x_batch, y_batch)


def generate_cv_batch(data_file, num_objects, batch_size, indx, mu_std):
    """
    NAME: generate_cv_batch
    PURPOSE: To generate training batch
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    while True:
        x_batch, y_batch = load_batch_from_h5(data_file, num_objects, batch_size, indx, mu_std)
        yield (x_batch, y_batch)
