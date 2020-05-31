# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import datetime
import os
import warnings

import tensorflow as tf
from tensorflow.python.platform.test import is_built_with_cuda, is_gpu_available


def cpu_fallback(flag=True):
    """
    A function to force Tensorflow to use CPU even Nvidia GPU present

    :param flag: True to fallback to CPU, False to un-manage CPU or GPU
    :type flag: bool
    :History:
        | 2017-Nov-25 - Written - Henry Leung (University of Toronto)
        | 2020-May-31 - Update for tf 2
    """
    gpu_phy_devices = tf.config.list_physical_devices('GPU')
    cpu_phy_devices = tf.config.list_physical_devices('CPU')

    general_warning_msg = "Tensorflow has already been initialized, this function needs to be called before any " \
                          "Tensorflow operation, as a result this function will have no effect"

    if flag is True:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            warnings.warn(general_warning_msg)
    elif flag is False:
        try:
            tf.config.set_visible_devices(gpu_phy_devices, 'GPU')
        except RuntimeError:
            warnings.warn(general_warning_msg)
    else:
        raise ValueError('Unknown flag, can only be True of False!')


def gpu_memory_manage(ratio=None, log_device_placement=False):
    """
    To manage GPU memory usage, prevent Tensorflow preoccupied all the video RAM

    :param ratio: Optional, ratio of GPU memory pre-allocating to astroNN
    :type ratio: Union[NoneType, float]
    :param log_device_placement: whether or not log the device placement
    :type log_device_placement: bool
    :History: 2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    config =  tf.compat.v1.ConfigProto()
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        if is_built_with_cuda():
            if ratio <= 0. or ratio > 1.:
                print(f"Invalid ratio argument -> ratio: {ratio}, it has been reset to ratio=1.0")
                ratio = 1.
            config.gpu_options.per_process_gpu_memory_fraction = ratio
        elif isinstance(ratio, float):
            warnings.warn("You have set GPU memory limit in astroNN config file but you are not using Tensorflow-GPU!")
    config.log_device_placement = log_device_placement

    if tf.compat.v1.get_default_session() is not None:
        warnings.warn("A Tensorflow session in use is detected, "
                      "astroNN will use that session to prevent overwriting session!")
    else:
        # Set global _SESSION for tensorflow to use with astroNN cpu, GPU setting
        tf.compat.v1.Session(config=config).__enter__()  # to register it as tensorflow default session

    return None


def gpu_availability():
    """
    Detect gpu on user system

    :return: Whether at least a CUDA compatible GPU is detected and usable
    :rtype: bool
    :History: 2018-Apr-25 - Written - Henry Leung (University of Toronto)
    """
    # assume if using tensorflow-gpu, then Nvidia GPU is available
    if is_built_with_cuda():
        return is_gpu_available()
    else:
        return is_built_with_cuda()


def folder_runnum():
    """
    To get the smallest available folder name without replacing the existing folder

    :return: folder name
    :rtype: str
    :History: 2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    now = datetime.datetime.now()
    runnum = 1
    while True:
        folder_name = f'astroNN_{now.month:0{2}d}{now.day:0{2}d}_run{runnum:0{3}d}'
        if not os.path.exists(folder_name):
            break
        else:
            runnum += 1

    return folder_name
