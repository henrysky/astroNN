# ---------------------------------------------------------------#
#   astroNN.nn.numpy: tools written with numpy instead of tf
# ---------------------------------------------------------------#
import numpy as np


def sigmoid(x):
    """
    NAME: sigmoid
    PURPOSE: numpy implementation of tf.sigmoid
    INPUT:
        x (ndarray): input
    OUTPUT:
        (ndarray)
    HISTORY:
        2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_inv(x):
    """
    NAME: sigmoid_inv
    PURPOSE: numpy implementation of tf.sigmoid inverse
    INPUT:
        x (ndarray): input
    OUTPUT:
        (ndarray)
    HISTORY:
        2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return np.log(x / (1 - x))


def l1(x, l1=0.):
    """
    NAME: l1
    PURPOSE: numpy implementation of tf.nn.l1
    INPUT:
        x (ndarray): input
    OUTPUT:
        (ndarray)
    HISTORY:
        2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    l1_x = 0.
    l1_x += np.sum(l1 * np.abs(x))
    return l1_x


def l2(x, l2=0.):
    """
    NAME: l2
    PURPOSE: numpy implementation of tf.nn.l2
    INPUT:
        x (ndarray): input
    OUTPUT:
        (ndarray) representing regularising term
    HISTORY:
        2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    l2_x = 0.
    l2_x += np.sum(l2 * np.square(x))
    return l2_x


def relu(x):
    """
    NAME: relu
    PURPOSE: numpy implementation of tf.nn.relu
    INPUT:
        x (ndarray): input
    OUTPUT:
        (ndarray) representing activated ndarray
    HISTORY:
        2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return x * (x > 0)
