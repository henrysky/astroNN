# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import os
import time
import numpy as np
import datetime

import tensorflow as tf
from keras.backend import learning_phase, function, set_session, clear_session


def h5name_check(h5name):
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5_filename="..."')
    return None

def cpu_fallback():
    """
    NAME:
        cpu_fallback
    PURPOSE:
        use CPU even Nvidia GPU present
    INPUT:
        None
    OUTPUT:
        None
    HISTORY:
        2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('astroNN will be using CPU, please ignore Tensorflow warning on PCIe device')


def gpu_memory_manage(ratio=None, log_device_placement=False):
    """
    NAME:
        gpu_memory_manage
    PURPOSE:
        to manage GPU memory usage, prevent Tensorflow preoccupied all the video RAM
    INPUT:
        ratio (float): Optional, ratio of GPU memory pre-allocating to astroNN
        log_device_placement (boolean): whether or not log the device placement
    OUTPUT:
        None
    HISTORY:
        2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    config = tf.ConfigProto()
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    config.log_device_placement = log_device_placement
    set_session(tf.Session(config=config))

    return None


def folder_runnum():
    now = datetime.datetime.now()
    folder_name = None
    for runnum in range(1, 99999):
        folder_name = 'astroNN_{}{:02d}_run{:03d}'.format(now.month, now.day, runnum)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        else:
            runnum += 1

    return folder_name


def denormalize(normalized, std_labels, mean_labels):
    """
    NAME:
        denormalize
    PURPOSE:
        to denormalize the normalize input from Neural Network
    INPUT:
        normalized (ndarray)
        std_labels (ndarray)
        mean_labels (ndarray)
    OUTPUT:
        (ndarray): denormalized array
    HISTORY:
        2017-Oct-01 - Written - Henry Leung (University of Toronto)
    """
    return (normalized * std_labels) + mean_labels


def batch_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels):
    total_spectra_num = spectra.shape[0]
    predictions = np.zeros((total_spectra_num, num_labels))
    start_time = time.time()

    for i in range(len(spectra) // batch_size):
        inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
        predictions[i * batch_size:(i + 1) * batch_size] = denormalize(model.predict(inputs), std_labels, mean_labels)
        print('Competed {} of {}, {:.03f}s Elapsed'.format((i + 1) * batch_size, total_spectra_num,
                                                           time.time() - start_time))

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


def batch_dropout_predictions(model, spectra, batch_size, num_labels, std_labels, mean_labels, mc_dropout_num=100):
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
    uncertainty_2_dropout = np.zeros((total_spectra_num, num_labels))
    uncertainty_master = np.zeros((total_spectra_num, num_labels))
    get_dropout_output = function([model.layers[0].input, learning_phase()], [model.layers[-1].output])

    print('\n')
    print('MC Dropout enabled')
    print('MC Dropout Prediction will probably take a long time')
    start_time = time.time()

    for i in range(len(spectra) // batch_size):
        predictions = np.zeros((mc_dropout_num, batch_size, num_labels))
        uncertainty = np.zeros((mc_dropout_num, batch_size, num_labels))
        for j in range(mc_dropout_num):
            inputs = spectra[i * batch_size:(i + 1) * batch_size].reshape((batch_size, spectra.shape[1], 1))
            result = get_dropout_output([inputs, 1])[0]
            predictions[j, :] = denormalize(result[:, :num_labels], std_labels, mean_labels)
            uncertainty[j, :] = denormalize(np.exp(result[:, num_labels:]), std_labels, np.zeros(std_labels.shape))
        print('Completed {} of {}, {:.03f} seconds elapsed'.format((i + 1) * batch_size, total_spectra_num,
                                                                   time.time() - start_time))
        prediction_mc_droout[i * batch_size:(i + 1) * batch_size] = np.mean(predictions, axis=0)
        uncertainty_mc_dropout[i * batch_size:(i + 1) * batch_size] = np.var(predictions, axis=0)
        uncertainty_2_dropout[i * batch_size:(i + 1) * batch_size] = np.mean(uncertainty, axis=0)

        uncertainty_master[i * batch_size:(i + 1) * batch_size] = \
            uncertainty_mc_dropout[i * batch_size:(i + 1) * batch_size] + uncertainty_2_dropout[
                                                                          i * batch_size:(i + 1) * batch_size]

    try:
        i
    except NameError:
        i = 0
        batch_size = 0

    if (i + 1) * batch_size != spectra.shape[0]:
        number = spectra.shape[0] - (i + 1) * batch_size
        predictions = np.zeros((mc_dropout_num, number, num_labels))
        uncertainty = np.zeros((mc_dropout_num, number, num_labels))
        for j in range(mc_dropout_num):
            inputs = spectra[(i + 1) * batch_size:].reshape((number, spectra.shape[1], 1))
            result = get_dropout_output([inputs, 1])[0]
            predictions[j, :] = denormalize(result[:, :num_labels], std_labels, mean_labels)
            uncertainty[j, :] = denormalize(result[:, num_labels:], std_labels, np.zeros(std_labels.shape))
        prediction_mc_droout[(i + 1) * batch_size:] = np.mean(predictions, axis=0)
        uncertainty_mc_dropout[(i + 1) * batch_size:] = np.var(predictions, axis=0)
        uncertainty_2_dropout[(i + 1) * batch_size:] = np.mean(uncertainty, axis=0)
        uncertainty_master[(i + 1) * batch_size:] = uncertainty_mc_dropout[(i + 1) * batch_size:] + uncertainty[(i + 1) * batch_size:]

    # tau = l ** 2 * (1 - model.p) / (2 * N * model.weight_decay)
    # uncertainty_mc_dropout += tau ** -1
    # TODO: Model Precision
    print('\n')
    clear_session()

    return prediction_mc_droout, uncertainty_master


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

