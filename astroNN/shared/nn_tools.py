# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import os
import numpy as np

import tensorflow as tf
from keras.backend import set_session
from keras.backend import learning_phase, function


def h5name_check(h5name):
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')
    return  None


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
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Nov-25 Henry Leung
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('astroNN will be using CPU, please ignore Tensorflow warning on PCIe device')


def gpu_memory_manage():
    """
    NAME: cpu_fallback
    PURPOSE: use cpu even gpu present
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Nov-25 Henry Leung
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def denormalize(lb_norm, std_labels, mean_labels):
    return (lb_norm * std_labels) + mean_labels


def batch_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels):
    predictions = np.zeros((len(spectra), num_labels))
    i = 0
    for i in range(len(spectra) // batch_size):
        inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
        predictions[i * batch_size:(i + 1) * batch_size] = denormalize(model.predict(inputs), std_labels, mean_labels)
    if (i + 1) * batch_size != len(spectra): # Prevet None size error if length is mulitpler of batch_size
        inputs = spectra[(i + 1) * batch_size:].reshape((spectra[(i + 1) * batch_size:].shape[0], spectra.shape[1], 1))
        predictions[(i + 1) * batch_size:] = denormalize(model.predict(inputs), std_labels, mean_labels)
    model_uncertainty = np.zeros(predictions.shape)
    return predictions, model_uncertainty


def batch_dropout_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels):
    predictions = np.zeros((len(spectra), num_labels))
    dropout_total = 50
    master_predictions = np.zeros((dropout_total, len(spectra), num_labels))
    i = 0
    get_dropout_output = function([model.layers[0].input, learning_phase()], [model.layers[-1].output])
    for j in range(dropout_total):
        for i in range(len(spectra) // batch_size):
            inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
            predictions[i * batch_size:(i + 1) * batch_size] = denormalize(get_dropout_output([inputs, 1])[0], std_labels, mean_labels)
        if (i + 1) * batch_size != len(spectra): # Prevet None size error if length is mulitpler of batch_size
            inputs = spectra[(i + 1) * batch_size:].reshape((spectra[(i + 1) * batch_size:].shape[0], spectra.shape[1], 1))
            predictions[(i + 1) * batch_size:] = denormalize(get_dropout_output([inputs, 1])[0], std_labels, mean_labels)
            master_predictions[j,:] = predictions

    prediction = np.mean(master_predictions, axis=0)
    model_uncertainty = np.std(master_predictions, axis=0) / np.sqrt(dropout_total)

    return prediction, model_uncertainty


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