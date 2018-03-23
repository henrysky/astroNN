# ---------------------------------------------------------------#
#   astroNN.models.utilities.metrics: metrics
# ---------------------------------------------------------------#

import tensorflow as tf

from astroNN.config import MAGIC_NUMBER
from astroNN.nn import magic_correction_term
from astroNN.nn.losses import mean_absolute_error
from astroNN.nn.losses import mean_absolute_percentage_error
from astroNN.nn.losses import mean_squared_error
from astroNN.nn.losses import mean_squared_logarithmic_error


# Just alias functions
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error


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


def binary_accuracy(from_logits=False):
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
    def binary_accuracy_internal(y_true, y_pred):
        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32), axis=-1) * magic_correction_term(
            y_true)
    return binary_accuracy_internal
