# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import datetime
import os
from keras.backend.common import global_state


# TODO: removed gpu_memory_manage() and gpu_availability() as they are not used in astroNN

def cpu_fallback(flag=True):
    """
    A function to force Tensorflow to use CPU even Nvidia GPU present

    :param flag: `True` to fallback to CPU, `False` to un-manage CPU or GPU
    :type flag: bool
    :History:
        | 2017-Nov-25 - Written - Henry Leung (University of Toronto)
        | 2020-May-31 - Update for tf 2
        | 2023-Dec-27 - Update for Keras 3.0
    """
    if flag is True:
        global_state.set_global_attribute("torch_device", "cpu")
    elif flag is False:
        global_state.set_global_attribute("torch_device", "cuda")
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
