# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import datetime
import os

import tensorflow as tf
from tensorflow.python.platform.test import is_built_with_cuda


def cpu_fallback(flag=0):
    """
    A function to force Tensorflow to use CPU even Nvidia GPU present

    :param flag: flag=0 to fallback to CPU, flag=1 to reverse the operation
    :type flag: int
    :History: 2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    if flag == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('astroNN is forced to use CPU as you have requested, please ignore Tensorflow CUDA_ERROR_NO_DEVICE '
              'warning!')
    elif flag == 1:
        try:
            del os.environ['CUDA_VISIBLE_DEVICES']
            print('astroNN will let Tensorflow to decide whether using CPU or GPU')
        except KeyError:
            pass
    else:
        raise ValueError('Unknown flag, it can only either be 0 or 1!')


def gpu_memory_manage(ratio=None, log_device_placement=False):
    """
    To manage GPU memory usage, prevent Tensorflow preoccupied all the video RAM

    :param ratio: Optional, ratio of GPU memory pre-allocating to astroNN
    :type ratio: Union[NoneType, float]
    :param log_device_placement: whether or not log the device placement
    :type log_device_placement: bool
    :History: 2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    config = tf.ConfigProto()
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        if ratio <= 0. or ratio > 1.:
            print(f"Invalid ratio argument -> ratio: {ratio}, it has been reset to ratio=1.0")
            ratio = 1.
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    config.log_device_placement = log_device_placement

    # Set global _SESSION for tensorflow to use with astroNN cpu, GPU setting
    tf.Session(config=config).__enter__()  # to register it as tensorflow default session

    return None


def gpu_availability():
    """
    Detect gpu on user system

    :return: Whether at least a CUDA compatible GPU is detected by assuming using tensorflow-gpu means CUDA GPU exists
    :rtype: bool
    :History: 2018-Apr-25 - Written - Henry Leung (University of Toronto)
    """
    # assume if using tensorflow-gpu, then Nvidia GPU is available
    return is_built_with_cuda()


def folder_runnum():
    """
    To get the smallest available folder name without replacing the existing folder

    :return: folder name
    :rtype: str
    :History: 2017-Nov-25 - Written - Henry Leung (University of Toronto)
    """
    now = datetime.datetime.now()
    folder_name = None
    runnum = 1
    while True:
        folder_name = f'astroNN_{now.month:0{2}d}{now.day:0{2}d}_run{runnum:0{3}d}'
        if not os.path.exists(folder_name):
            break
        else:
            runnum += 1

    return folder_name
