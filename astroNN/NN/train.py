# ---------------------------------------------------------#
#   astroNN.NN.train: train models
# ---------------------------------------------------------#

import h5py
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session
import astroNN.NN.cnn_models
import astroNN.NN.train_tools
import astroNN.NN.test
from keras.utils import plot_model
import os
import datetime


def apogee_train(h5name=None, target=None, test=True, model=None):
    """
    NAME: apogee_train
    PURPOSE: To train
    INPUT:
        h5name: name of h5 data, {h5name}_train.h5   {h5name}_test.h5
        target name (list):
                spec
                SNR
                RA
                DEC
                teff
                logg
                MH
                alpha_M
                C
                Cl
                N
                O
                Na
                Mg
                Al
                Si
                Ca
                Ti
                Ti2
                Fe
                Ni
        test (boolean): whether test data or not after training
        model: which model defined in astroNN.NN.cnn_model.py
    OUTPUT: target and normalized data
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="...... "')
    if target is None:
        raise ValueError('Please specift a list of target names using target=[.., ...], target must be a list')
    if model is None:
        model = 'cnn_apogee_1'
        print('No predefined model specified, using cnn_apogee_1 as default')

    target = np.asarray(target)
    h5data = h5name + '_train.h5'
    h5test = h5name + '_test.h5'

    num_labels = target.shape

    with h5py.File(h5data) as F:
        spectra = F['spectra']
        num_flux = spectra.shape[1]
        num_train = int(0.9 * spectra.shape[0])  # number of training example, rest are cross validation
        num_cv = spectra.shape[0] - num_train  # cross validation
    print('Each spectrum contains ' + str(num_flux) + ' wavelength bins')
    print('Training set includes ' + str(num_train) + ' spectra and the cross-validation set includes ' + str(num_cv)
          + ' spectra')

    # load data
    F = h5py.File(h5data, 'r')
    X = F['spectra']
    y = np.array((X.shape[1]))
    i = 0
    mean_labels = np.array([])
    std_labels = np.array([])
    model_name = ''
    for tg in target:
        temp = F['{}'.format(tg)]
        mean_labels = np.append(mean_labels, np.mean(temp))
        std_labels = np.append(std_labels, np.std(temp))
        model_name = model_name + '_{}'.format(tg)

    mu_std = np.vstack((mean_labels, std_labels))
    num_labels = mu_std.shape[1]
    # prevent Tensorflow taking up all the GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # activation function used following every layer except for the output layers
    activation = 'relu'

    # model weight initializer
    initializer = 'he_normal'

    # shape of input spectra that is fed into the input layer
    input_shape = (None, num_flux, 1)

    # number of filters used in the convolutional layers
    num_filters = [2, 4]

    # length of the filters in the convolutional layers
    filter_length = 8

    # length of the maxpooling window
    pool_length = 4

    # number of nodes in each of the hidden fully connected layers
    num_hidden = [128, 64]

    # number of spectra fed into model at once during training
    batch_size = 64

    # maximum number of interations for model training
    max_epochs = 5

    # initial learning rate for optimization algorithm
    lr = 0.00007

    # exponential decay rate for the 1st moment estimates for optimization algorithm
    beta_1 = 0.9

    # exponential decay rate for the 2nd moment estimates for optimization algorithm
    beta_2 = 0.999

    # a small constant for numerical stability for optimization algorithm
    optimizer_epsilon = 1e-08

    # model selection according to user-choice
    model = getattr(astroNN.NN.cnn_models, model)(input_shape, initializer, activation, num_filters, filter_length,
                                               pool_length, num_hidden, num_labels)

    # Default loss function parameters
    early_stopping_min_delta = 0.0001
    early_stopping_patience = 4
    reduce_lr_factor = 0.5
    reuce_lr_epsilon = 0.0009
    reduce_lr_patience = 2
    reduce_lr_min = 0.00008
    loss_function = 'mean_squared_error'

    # compute accuracy and mean absolute deviation
    metrics = ['accuracy', 'mae']

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                   patience=early_stopping_patience, verbose=2, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    history = model.fit_generator(
        astroNN.NN.train_tools.generate_train_batch(F, num_train, batch_size, 0, mu_std, target),
        steps_per_epoch=num_train / batch_size,
        epochs=max_epochs,
        validation_data=astroNN.NN.train_tools.generate_cv_batch(F, num_cv, batch_size,
                                                                 num_train, mu_std, target),
        max_queue_size=10, verbose=2,
        callbacks=[early_stopping, reduce_lr],
        validation_steps=num_cv / batch_size)

    now = datetime.datetime.now()
    folder_name = 'apogee_train_{}{:02d}{}'.format(now.month, now.day, model_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    folder_name = folder_name + '\\'
    currentdir = os.getcwd()
    fullfilepath = os.path.join(currentdir, folder_name)

    starnet_model = 'cnn_{}.h5'.format(model_name)
    model.save(folder_name + starnet_model)
    print(starnet_model + ' saved to {}'.format(fullfilepath))
    print(model.summary())
    numpy_loss_history = np.array(history)
    np.save(folder_name + 'meanstd_starnet.npy', mu_std)
    np.save(folder_name + 'targetname.npy', target)
    plot_model(model,show_shapes=True,
               to_file=folder_name + 'apogee_train_{}{:02d}{}.png'.format(now.month, now.day, model_name))

    if test is True:
        astroNN.NN.test.apogee_test(model=folder_name + starnet_model, testdata=h5test, folder_name=folder_name)

    return None
