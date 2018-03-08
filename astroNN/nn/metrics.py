# ---------------------------------------------------------------#
#   astroNN.models.utilities.metrics: metrics
# ---------------------------------------------------------------#

import tensorflow as tf

from astroNN import MAGIC_NUMBER
from astroNN.nn import magic_correction_term
from astroNN.nn.losses import mae
from astroNN.nn.losses import mape
from astroNN.nn.losses import mse
from astroNN.nn.losses import msle


def mean_squared_error(*args):
    # Just a alias function
    return mse(*args)


def mean_absolute_error(*args):
    # Just a alias function
    return mae(*args)


def mean_squared_logarithmic_error(*args):
    # Just a alias function
    return msle(*args)


def mean_absolute_percentage_error(*args):
    # Just a alias function
    return mape(*args)


def MSE(*args):
    # Just a alias function
    return mse(*args)


def MAE(*args):
    # Just a alias function
    return mae(*args)


def MSLE(*args):
    # Just a alias function
    return msle(*args)


def MAPE(*args):
    # Just a alias function
    return mape(*args)


def categorical_accuracy(y_true, y_pred):
    """
    NAME: categorical_accuracy
    PURPOSE: Calculate categorical accuracy
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-21 - Written - Henry Leung (University of Toronto)
    """
    y_true = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), y_true)
    return tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)),
                   tf.float32) * magic_correction_term(y_true)


def binary_accuracy(y_true, y_pred):
    """
    NAME: binary_accuracy
    PURPOSE: Calculate binary accuracy
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-21 - Written - Henry Leung (University of Toronto)
    """
    # DO NOT correct y_true for magic number, just let it goes wrong and then times a correction terms
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32), axis=-1) * magic_correction_term(
        y_true)
