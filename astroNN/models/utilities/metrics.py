# ---------------------------------------------------------------#
#   astroNN.models.utilities.metrics: metrics
# ---------------------------------------------------------------#

import keras.backend as K
from astroNN import MAGIC_NUMBER
from astroNN.models.losses.regression import mean_absolute_error as mae


def mean_absolute_error(*args):
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
    y_true = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), y_true)
    return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)),
                  K.floatx())


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
    y_true = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), y_true)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
