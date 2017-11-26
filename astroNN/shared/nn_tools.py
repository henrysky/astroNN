# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#
import os
import tensorflow as tf
from keras.backend import set_session


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('astroNN will be using CPU, please ignore Tensorflow warning on PCIe device')


def gpu_memory_manage():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))