# ---------------------------------------------------------#
#   astroNN.NN.generative: train generative models
# ---------------------------------------------------------#

import datetime
import os
import random

import h5py
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.utils import plot_model

import astroNN.NN.cnn_models
import astroNN.NN.generative_test
import astroNN.NN.train_tools


def apogee_generative_train(h5name=None, model=None, test=False):
    """
    NAME: apogee_generative_train
    PURPOSE: To train generative model
    INPUT:
        h5name: name of h5 data, {h5name}_train.h5
        model: which model defined in astroNN.NN.cnn_model.py
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="...... "')

    if model is None:
        model = 'apogee_generative_1'
        print('No predefined model specified, using apogee_generative_1 as default')

    h5data = h5name + '_train.h5'
    h5test = h5name + '_test.h5'

    with h5py.File(h5data) as F:  # ensure the file will be cleaned up
        spectra = np.array(F['spectra'])
        y = np.array(F['spectrabestfit'])
        num_flux = spectra.shape[1]
        input_std = spectra.std()
        output_std = spectra.std()
        spectra -= 1
        spectra /= input_std
        y -= 1
        y /= output_std
        num_train = int(0.9 * spectra.shape[0])  # number of training example, rest are cross validation
        num_cv = spectra.shape[0] - num_train  # cross validation
        model_name = 'generative'

    print('Each spectrum contains ' + str(num_flux) + ' wavelength bins')
    print('Training set includes ' + str(num_train) + ' spectra and the cross-validation set includes ' + str(num_cv)
          + ' spectra')

    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    activation = 'relu'  # activation function used following every layer except for the output layers
    initializer = 'he_normal'  # model weight initializer
    input_shape = (None, num_flux, 1)  # shape of input spectra that is fed into the input layer
    num_hidden = [32, 16, 32]  # number of nodes in each of the hidden fully connected layers
    batch_size = 64  # number of spectra fed into model at once during training
    max_epochs = 10  # maximum number of interations for model training
    lr = 0.00000007  # initial learning rate for optimization algorithm
    beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
    beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
    optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm

    # model selection according to user-choice
    model = getattr(astroNN.NN.cnn_models, model)(input_shape, initializer, activation, num_hidden)

    # Default loss function parameters
    early_stopping_min_delta = 0.00001
    early_stopping_patience = 15
    reuce_lr_epsilon = 0.009
    reduce_lr_patience = 2
    reduce_lr_min = 0.00000000001
    loss_function = 'mean_squared_error'

    # compute accuracy and mean absolute deviation
    metrics = ['mae']

    now = datetime.datetime.now()
    for runno in range(1, 99999):
        folder_name = 'apogee_train_{}{:02d}_run{}{}'.format(now.month, now.day, runno, model_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        else:
            runno += 1
    folder_name = folder_name + '/'
    currentdir = os.getcwd()
    fullfilepath = os.path.join(currentdir, folder_name)

    csv_logger = CSVLogger(fullfilepath + 'log.csv', append=True, separator=',')

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                   patience=early_stopping_patience, verbose=2, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    model.fit_generator(astroNN.NN.generative.generate_train_batch(num_train, batch_size, 0, spectra, y),
                        steps_per_epoch=num_train / batch_size,
                        epochs=max_epochs,
                        validation_data=astroNN.NN.generative.generate_cv_batch(num_cv, batch_size, num_train, spectra,
                                                                                y),
                        max_queue_size=10, verbose=2, callbacks=[early_stopping, reduce_lr, csv_logger],
                        validation_steps=num_cv / batch_size)

    astronn_model = 'generative_{}.h5'.format(model_name)
    model.save(folder_name + astronn_model)
    print(astronn_model + ' saved to {}'.format(fullfilepath))
    print(model.summary())
    plot_model(model, show_shapes=True,
               to_file=folder_name + 'apogee_train_{}{:02d}{}.png'.format(now.month, now.day, model_name))

    # Test after training
    if test is True:
        astroNN.NN.generative_test.apogee_generative_test(model=folder_name + astronn_model, testdata=h5test,
                                                          folder_name=folder_name, std=[input_std, output_std])
    return None


def load_batch(num_train, batch_size, indx, spectra, y):
    # Generate list of random indices (within the relevant partition of the main data file, e.g. the
    # training set) to be used to index into data_file
    indices = random.sample(range(indx, indx + num_train), batch_size)
    indices = np.sort(indices)

    # load data
    spectra = spectra[indices, :]
    normed_y = y[:][indices]

    # Reshape X data for compatibility with CNN
    spectra = spectra.reshape(len(spectra), spectra.shape[1], 1)

    return spectra, normed_y


def generate_train_batch(num_objects, batch_size, indx, spectra, y):
    while True:
        x_batch, y_batch = load_batch(num_objects, batch_size, indx, spectra, y)
        yield (x_batch, y_batch)


def generate_cv_batch(num_objects, batch_size, indx, spectra, y):
    while True:
        x_batch, y_batch = load_batch(num_objects, batch_size, indx, spectra, y)
        yield (x_batch, y_batch)
