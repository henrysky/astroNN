# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import os

import numpy as np
import time

import tensorflow as tf
from keras.backend import learning_phase, function, set_session, clear_session


def h5name_check(h5name):
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')
    return None


def foldername_modelname(folder_name=None):
    """
    NAME: foldername_modelname
    PURPOSE: convert foldername to model name
    INPUT:
        folder_name = folder name
    OUTPUT: model name
    HISTORY:
        2017-Nov-20 Henry Leung
    """
    return '/model_{}.h5'.format(folder_name[-11:])


def cpu_fallback():
    """
    NAME: cpu_fallback
    PURPOSE: use cpu even gpu present
    INPUT: None
    OUTPUT: None
    HISTORY:
        2017-Nov-25 Henry Leung
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('astroNN will be using CPU, please ignore Tensorflow warning on PCIe device')


def gpu_memory_manage():
    """
    NAME: gpu_memory_manage
    PURPOSE: to manage GPU memory usage, prevent Tensorflow preoccupied all the video RAM
    INPUT: None
    OUTPUT: None
    HISTORY:
        2017-Nov-25 Henry Leung
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def denormalize(lb_norm, std_labels, mean_labels):
    """
    NAME: denormalize
    PURPOSE: to denormalize the normalize input from Neural Network
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Oct-01 Henry Leung
    """
    return (lb_norm * std_labels) + mean_labels


def batch_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels):
    total_spectra_num = spectra.shape[0]
    predictions = np.zeros((total_spectra_num, num_labels))
    start_time = time.time()

    for i in range(len(spectra) // batch_size):
        inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
        predictions[i * batch_size:(i + 1) * batch_size] = denormalize(model.predict(inputs), std_labels, mean_labels)
        print('Competed {} of {}, {:.03f}s Elapsed'.format((i + 1) * batch_size, total_spectra_num, time.time()-start_time))


    try:
        i
    except NameError:
        i = 0
        batch_size = 0

    if (i + 1) * batch_size != spectra.shape[0]:
        number = spectra.shape[0] - (i + 1) * batch_size
        inputs = spectra[(i + 1) * batch_size:].reshape((number, spectra.shape[1], 1))
        predictions[:] = denormalize(model.predict(inputs), std_labels, mean_labels)

    model_uncertainty = np.zeros(predictions.shape)  # Zero model uncertainrt for traditional dropout

    print('\n')
    clear_session()

    return predictions, model_uncertainty


def batch_dropout_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels, mc_dropout_num=200):
    """
    NAME: batch_dropout_predictions
    PURPOSE: to use MC dropout to do model prediction
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Nov-28 Henry Leung
    """
    total_spectra_num = spectra.shape[0]
    prediction_mc_droout = np.zeros((total_spectra_num, num_labels))
    uncertainty_mc_dropout = np.zeros((total_spectra_num, num_labels))
    get_dropout_output = function([model.layers[0].input, learning_phase()], [model.layers[-1].output])

    print('\n')
    print('MC dropout Predicition will probably take a long time')
    start_time = time.time()

    for i in range(len(spectra) // batch_size):
        predictions = np.zeros((mc_dropout_num, batch_size, num_labels))
        for j in range(mc_dropout_num):
            inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
            predictions[j,:] = denormalize(get_dropout_output([inputs, 1])[0], std_labels, mean_labels)
        print('Competed {} of {}, {:.03f}s Elapsed'.format((i + 1) * batch_size, total_spectra_num, time.time()-start_time))
        prediction_mc_droout[i * batch_size:(i + 1) * batch_size] = np.mean(predictions, axis=0)
        uncertainty_mc_dropout[i * batch_size:(i + 1) * batch_size] = np.std(predictions, axis=0)

    try:
        i
    except NameError:
        i = 0
        batch_size = 0

    if (i + 1) * batch_size != spectra.shape[0]:
        number = spectra.shape[0] - (i + 1) * batch_size
        predictions = np.zeros((mc_dropout_num, number, num_labels))
        for j in range(mc_dropout_num):
            inputs = spectra[(i + 1) * batch_size:].reshape((number, spectra.shape[1], 1))
            predictions[j, :] = denormalize(get_dropout_output([inputs, 1])[0], std_labels, mean_labels)
        prediction_mc_droout[(i + 1) * batch_size:] = np.mean(predictions, axis=0)
        uncertainty_mc_dropout[(i + 1) * batch_size:] = np.std(predictions, axis=0)

    print('\n')
    clear_session()

    return prediction_mc_droout, uncertainty_mc_dropout


def target_name_conversion(targetname):
    if len(targetname) < 3:
        fullname = '[{}/H]'.format(targetname)
    elif targetname == 'teff':
        fullname = '$T_{\mathrm{eff}}$'
    elif targetname == 'alpha':
        fullname = '[Alpha/M]'
    elif targetname == 'logg':
        fullname = '[Log(g)]'
    elif targetname == 'Ti2':
        fullname = 'TiII'
    elif targetname == 'C1':
        fullname = 'CI'
    elif targetname == 'Cl':
        fullname = 'CI'
    else:
        fullname = targetname
    return fullname


def aspcap_windows_url_correction(targetname):
    if len(targetname) < 2:
        fullname = '{}'.format(targetname)
    elif targetname == 'teff':
        fullname = '$T_{\mathrm{eff}}$'
    elif targetname == 'alpha':
        fullname = '[Alpha/M]'
    elif targetname == 'logg':
        fullname = '[Log(g)]'
    elif targetname == 'Ti2':
        fullname = 'TiII'
    elif targetname == 'Cl':
        fullname = 'CI'
    else:
        fullname = targetname
    return fullname


def target_to_aspcap_conversion(targetname):
    if targetname == 'alpha':
        fullname = targetname + '_M'
    elif len(targetname) < 3:
        fullname = targetname + '_H'
    else:
        fullname = targetname
    return fullname