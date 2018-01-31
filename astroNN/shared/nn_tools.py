# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import datetime
import os

import tensorflow as tf
from keras.backend import set_session


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
    print('astroNN is forced to use CPU as you have requested, please ignore Tensorflow warning on PCIe device')


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
    """
    NAME:
        folder_runnum
    PURPOSE:
        to get the smallest available folder name without replacing the existing folder
    INPUT:
        None
    OUTPUT:
        None
    HISTORY:
        2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    now = datetime.datetime.now()
    folder_name = None
    for runnum in range(1, 99999):
        folder_name = 'astroNN_{:02d}{:02d}_run{:03d}'.format(now.month, now.day, runnum)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        else:
            runnum += 1

    return folder_name


def target_name_conversion(targetname):
    """
    NAME:
        target_name_conversion
    PURPOSE:
        to convert targetname to string used to plot graph
    INPUT:
        None
    OUTPUT:
        None
    HISTORY:
        2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    if targetname == 'C1':
        fullname = 'CI'
    elif len(targetname) < 3:
        fullname = '[{}/H]'.format(targetname)
    elif targetname == 'teff':
        fullname = '$T_{\mathrm{eff}}$'
    elif targetname == 'alpha':
        fullname = '[Alpha/M]'
    elif targetname == 'logg':
        fullname = '[Log(g)]'
    elif targetname == 'Ti2':
        fullname = 'TiII'
    else:
        fullname = targetname
    return fullname


def aspcap_windows_url_correction(targetname):
    """
    NAME:
        target_name_conversion
    PURPOSE:
        to convert targetname to string used to get ASPCAP windows url
    INPUT:
        None
    OUTPUT:
        None
    HISTORY:
        2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    if targetname == 'C1':
        fullname = 'CI'
    elif len(targetname) < 3:
        fullname = '{}'.format(targetname)
    elif targetname == 'teff':
        fullname = 'Surface Temperature'
    elif targetname == 'alpha':
        fullname = '[Alpha/M]'
    elif targetname == 'logg':
        fullname = '[Log(g)]'
    elif targetname == 'Ti2':
        fullname = 'TiII'
    else:
        fullname = targetname
    return fullname
