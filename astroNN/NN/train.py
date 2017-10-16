# ---------------------------------------------------------#
#   astroNN.NN.train: train models
# ---------------------------------------------------------#

import h5py
import random
import numpy as np
import astroNN.NN.cnn_models


def apogee_train(h5data=None, target=None, h5test=None, num_filter=[4, 16]):
    """
    NAME: load_batch_from_h5
    PURPOSE: To load batch for training from .h5 dataset file
    INPUT:
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    if h5data is None:
        raise ('Please specift the dataset name using h5data="......"')
    if target is None:
        raise ('Please specift a list of target names using target=[.., ...]')
    if h5test is None:
        raise ('Please specift the testset name using h5test="......"')

    num_labels = target.shape

    with h5py.File(h5data) as F:
        spectra = F['spectra']
        num_flux = spectra.shape[1]
        num_train = int(0.9*spectra.shape[0])
        num_cv = spectra.shape[0] - num_train # cross validation
    print('Each spectrum contains ' + str(num_flux) + ' wavelength bins')
    print('Training set includes ' + str(num_train) + ' spectra and the cross-validation set includes ' + str(num_cv) + ' spectra')

    # activation function used following every layer except for the output layers
    activation = 'relu'

    # model weight initializer
    initializer = 'he_normal'

    # shape of input spectra that is fed into the input layer
    input_shape = (None, num_flux, 1)

    # number of filters used in the convolutional layers
    num_filters = num_filter

    # length of the filters in the convolutional layers
    filter_length = 8

    # length of the maxpooling window
    pool_length = 4

    # number of nodes in each of the hidden fully connected layers
    num_hidden = [256, 128]

    # number of spectra fed into model at once during training
    batch_size = 64

    # maximum number of interations for model training
    max_epochs = 30

    # initial learning rate for optimization algorithm
    lr = 0.0007

    # exponential decay rate for the 1st moment estimates for optimization algorithm
    beta_1 = 0.9

    # exponential decay rate for the 2nd moment estimates for optimization algorithm
    beta_2 = 0.999

    # a small constant for numerical stability for optimization algorithm
    optimizer_epsilon = 1e-08

    astroNN.NN.cnn_models.cnn_model_1(input_shape, initializer, activation, num_filters, filter_length, pool_length,
                                      num_hidden, num_labels)

    return None


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
