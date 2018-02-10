# ---------------------------------------------------------------#
#   astroNN.models.losses.regression: losses function for regression
# ---------------------------------------------------------------#
import tensorflow as tf

from astroNN import MAGIC_NUMBER


def magic_correction_term(y_true):
    """
    NAME: magic_correction_term
    PURPOSE: calculate a correction term to prevent the loss being lowered by magic_num, since we assume
    whatever neural network is predicting, its right for those magic number
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-30 - Written - Henry Leung (University of Toronto)
    """
    # Get a mask with those -9999.
    mask = tf.equal(y_true, MAGIC_NUMBER)
    num_nonzero = tf.cast(tf.count_nonzero(mask, axis=-1), tf.float32)
    num_zero = tf.cast(tf.reduce_sum(tf.to_int32(mask), reduction_indices=-1), tf.float32)

    # If no magic number, then num_zero=0 and whole expression is just 1 and get back our good old loss
    # If num_nonzero is 0, that means we don't have any information, then set the correction term to ones
    return tf.where(tf.equal(num_nonzero, 0), tf.zeros_like(num_zero), (num_nonzero + num_zero) / num_nonzero)


def mean_squared_error(y_true, y_pred):
    """
    NAME: mean_squared_error
    PURPOSE: calculate mean square error losses
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    return tf.reduce_mean(tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                                   tf.square(y_true - y_pred)), axis=-1)


def mse_lin_wrapper(var, labels_err):
    """
    NAME: mse_lin_wrapper
    PURPOSE: losses function for regression node in Bayesian Neural Network
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_lin(y_true, y_pred):
        # labels_err still contains magic_number
        labels_err_y = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), labels_err)
        # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
        y_pred_corrected = tf.log(tf.exp(var) + tf.square(labels_err_y))

        wrapper_output = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                                  0.5 * tf.square(y_true - y_pred) * (tf.exp(-y_pred_corrected)) + 0.5 *
                                  y_pred_corrected)

        return tf.reduce_mean(wrapper_output, axis=-1)

    return mse_lin


def mse_var_wrapper(lin, labels_err):
    """
    NAME: mse_var_wrapper
    PURPOSE: calculate predictive variance, and takes account of labels error  in Bayesian Neural Network
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-19 - Written - Henry Leung (University of Toronto)
    """

    def mse_var(y_true, y_pred):
        # labels_err still contains magic_number
        labels_err_y = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), labels_err)
        # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
        y_pred_corrected = tf.log(tf.exp(y_pred) + tf.square(labels_err_y))

        wrapper_output = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                                  0.5 * tf.square(y_true - lin) * (tf.exp(-y_pred_corrected)) + 0.5 *
                                  y_pred_corrected)

        return tf.reduce_mean(wrapper_output, axis=-1)

    return mse_var


def mean_absolute_error(y_true, y_pred):
    """
    NAME: mean_absolute_error
    PURPOSE: calculate mean absolute error, ignoring the magic number
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    return tf.reduce_mean(tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                                   tf.abs(y_true - y_pred)), axis=-1) * magic_correction_term(y_true)
