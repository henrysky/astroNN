# ---------------------------------------------------------------#
#   astroNN.models.utilities.metrics: metrics
# ---------------------------------------------------------------#

import tensorflow as tf

from astroNN import MAGIC_NUMBER
from astroNN.nn.losses import mean_absolute_error as mae


def mean_absolute_error(*args):
    # Just a alias function
    return mae(*args)


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
    return tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), 'float32')


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
    return tf.reduce_mean(tf.equal(y_true, tf.round(y_pred)), axis=-1)
