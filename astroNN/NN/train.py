# ---------------------------------------------------------#
#   astroNN.NN.train: train models
# ---------------------------------------------------------#

import h5py
import random
import numpy as np
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
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

    model = astroNN.NN.cnn_models.cnn_model_1(input_shape, initializer, activation, num_filters, filter_length,
                                              pool_length, num_hidden, num_labels)

    # loss function to minimize
    loss_function = 'mean_squared_error'

    # compute accuracy and mean absolute deviation
    metrics = ['accuracy', 'mae']

    tbCallBack = TensorBoard(log_dir=datadir + 'logs', histogram_freq=0, batch_size=32, write_graph=True,
                             write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None)

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                   patience=early_stopping_patience, verbose=2, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    model.fit_generator(generate_train_batch(datadir + training_set,
                                             num_train, batch_size, 0,
                                             datadir + normalization_data),
                        steps_per_epoch=num_train / batch_size,
                        epochs=max_epochs,
                        validation_data=generate_cv_batch(datadir + training_set,
                                                          num_cv, batch_size, num_train,
                                                          datadir + normalization_data),
                        max_q_size=10, verbose=2,
                        callbacks=[early_stopping, reduce_lr, tbCallBack],
                        validation_steps=num_cv / batch_size)

    starnet_model = 'cnn_temperature.h5'
    model.save(datadir + starnet_model)
    print(starnet_model + ' saved.')

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
