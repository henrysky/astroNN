# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import datetime
import os
from astroNN.config import keras
import inspect
import warnings
from astroNN.config import _KERAS_BACKEND

# TODO: removed gpu_memory_manage() and gpu_availability() as they are not used in astroNN


def cpu_fallback(flag=True):
    """
    A function to force Keras backend to use CPU even Nvidia GPU is presented

    :param flag: `True` to fallback to CPU, `False` to un-manage CPU or GPU
    :type flag: bool
    :History:
        | 2017-Nov-25 - Written - Henry Leung (University of Toronto)
        | 2020-May-31 - Update for tf 2
        | 2023-Dec-27 - Update for Keras 3.0
    """

    general_tf_warning_msg = (
        f"Tensorflow has already been initialized, {inspect.currentframe().f_code.co_name}() needs "
        f"to be called before any Tensorflow operation, as a result this function will have no effect"
    )

    if flag is True:
        if _KERAS_BACKEND == "torch":
            keras.backend.common.global_state.set_global_attribute("torch_device", "cpu")
        elif _KERAS_BACKEND == "tensorflow":
            import tensorflow as tf
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError:
                warnings.warn(general_tf_warning_msg)
    elif flag is False:
        if _KERAS_BACKEND == "torch":
            keras.backend.common.global_state.set_global_attribute("torch_device", "cuda")
        elif _KERAS_BACKEND == "tensorflow":
            import tensorflow as tf
            try:
                gpu_phy_devices = tf.config.list_physical_devices("GPU")
                tf.config.set_visible_devices(gpu_phy_devices, "GPU")
            except RuntimeError:
                warnings.warn(general_tf_warning_msg)
    else:
        raise ValueError("Unknown flag, can only be True of False!")


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
