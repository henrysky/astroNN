# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import os

import numpy as np
import tensorflow as tf
from keras.backend import learning_phase, function
from keras.backend import set_session


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
    predictions = np.zeros((len(spectra), num_labels))
    i = 0

    for i in range(len(spectra) // batch_size):
        inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
        predictions[i * batch_size:(i + 1) * batch_size] = denormalize(model.predict(inputs), std_labels, mean_labels)

    if not i:
        i = 0
        batch_size = 0

    if (i + 1) * batch_size != spectra.shape[0]:
        number = spectra.shape[0] - (i + 1) * batch_size
        inputs = spectra[(i + 1) * batch_size:].reshape((number, spectra.shape[1], 1))
        predictions[:] = denormalize(model.predict(inputs), std_labels, mean_labels)

    model_uncertainty = np.zeros(predictions.shape)

    return predictions, model_uncertainty


def batch_dropout_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels):
    dropout_total = 1000
    prediction_mc_droout = np.zeros((spectra.shape[0], num_labels))
    uncertainty_mc_dropout = np.zeros((spectra.shape[0], num_labels))
    i = 0
    get_dropout_output = function([model.layers[0].input, learning_phase()], [model.layers[-1].output])

    print(spectra.shape[0])
    print('\n')

    for i in range(len(spectra) // batch_size):
        predictions = np.zeros((dropout_total, batch_size, num_labels))
        print('i am doing the job')
        for j in range(dropout_total):
            inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
            predictions[j,:] = denormalize(get_dropout_output([inputs, 1])[0], std_labels, mean_labels)
        prediction_mc_droout[i * batch_size:(i + 1) * batch_size] = np.median(predictions, axis=0)
        uncertainty_mc_dropout[i * batch_size:(i + 1) * batch_size] = np.std(predictions, axis=0)

    if not i:
        i = 0
        batch_size = 0

    if (i + 1) * batch_size != spectra.shape[0]:
        number = spectra.shape[0] - (i + 1) * batch_size
        predictions = np.zeros((dropout_total, number, num_labels))
        for j in range(dropout_total):
            inputs = spectra[(i + 1) * batch_size:].reshape((number, spectra.shape[1], 1))
            predictions[j, :] = denormalize(get_dropout_output([inputs, 1])[0], std_labels, mean_labels)
        prediction_mc_droout[(i + 1) * batch_size:] = np.median(predictions, axis=0)
        uncertainty_mc_dropout[(i + 1) * batch_size:] = np.std(predictions, axis=0)

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