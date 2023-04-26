# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import datetime
import os
import inspect
import warnings

import tensorflow as tf
from tensorflow.python.platform.test import is_built_with_cuda


def cpu_fallback(flag=True):
    """
    A function to force Tensorflow to use CPU even Nvidia GPU present

    :param flag: True to fallback to CPU, False to un-manage CPU or GPU
    :type flag: bool
    :History:
        | 2017-Nov-25 - Written - Henry Leung (University of Toronto)
        | 2020-May-31 - Update for tf 2
    """
    gpu_phy_devices = tf.config.list_physical_devices("GPU")
    cpu_phy_devices = tf.config.list_physical_devices("CPU")

    general_warning_msg = (
        f"Tensorflow has already been initialized, {inspect.currentframe().f_code.co_name}() needs "
        f"to be called before any Tensorflow operation, as a result this function will have no effect"
    )

    if flag is True:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            warnings.warn(general_warning_msg)
    elif flag is False:
        try:
            tf.config.set_visible_devices(gpu_phy_devices, "GPU")
        except RuntimeError:
            warnings.warn(general_warning_msg)
    else:
        raise ValueError("Unknown flag, can only be True of False!")


def gpu_memory_manage(ratio=True, log_device_placement=False):
    """
    To manage GPU memory usage, prevent Tensorflow preoccupied all the video RAM

    :param ratio: Optional, ratio of GPU memory pre-allocating to astroNN
    :type ratio: Union[NoneType, float]
    :param log_device_placement: whether or not log the device placement
    :type log_device_placement: bool
    :History:
        | 2017-Nov-25 - Written - Henry Leung (University of Toronto)
        | 2020-Jun-1 - Updated for tf v2
    """
    gpu_phy_devices = tf.config.list_physical_devices("GPU")

    general_warning_msg = (
        f"Tensorflow has already been initialized, {inspect.currentframe().f_code.co_name}() needs "
        f"to be called before any Tensorflow operation, as a result this function will have no effect"
    )

    try:
        if ratio:
            for gpu in gpu_phy_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            for gpu in gpu_phy_devices:
                tf.config.experimental.set_memory_growth(gpu, False)

        if log_device_placement:
            tf.debugging.set_log_device_placement(True)
        else:
            tf.debugging.set_log_device_placement(False)
    except RuntimeError:
        warnings.warn(general_warning_msg)

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
        return len(tf.config.list_physical_devices("GPU")) > 0
    else:
        return False


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
        folder_name = f"astroNN_{now.month:0{2}d}{now.day:0{2}d}_run{runnum:0{3}d}"
        if not os.path.exists(folder_name):
            break
        else:
            runnum += 1

    return folder_name
