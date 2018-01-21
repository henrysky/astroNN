# ---------------------------------------------------------------#
#   astroNN.models.losses.regression: losses function for regression
# ---------------------------------------------------------------#
import keras.backend as K
from astroNN import MAGIC_NUMBER


def mean_squared_error(y_true, y_pred):
    """
    NAME: mean_squared_error
    PURPOSE: calculate mean square error losses
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    return K.mean(K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), K.square(y_true - y_pred)),
                  axis=-1)


def mse_lin_wrapper(var, labels_err):
    """
    NAME: mse_lin_wrapper
    PURPOSE: losses function for regression node
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_lin(y_true, y_pred):
        # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
        labels_err_x = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), labels_err)
        y_pred_corrected = K.log(K.exp(var) + K.square(labels_err_x))
        wrapper_output = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true),
                                    0.5 * K.square(y_true - y_pred) * (K.exp(-y_pred_corrected)) + 0.5 * y_pred_corrected)
        return K.mean(wrapper_output, axis=-1)

    return mse_lin


def mse_var_wrapper(lin):
    """
    NAME: mse_var_wrapper
    PURPOSE: calculate predictive variance
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_var(y_true, y_pred):
        wrapper_output = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true),
                                    0.5 * K.square(y_true - lin) * (K.exp(-y_pred)) + 0.5 * y_pred)
        return K.mean(wrapper_output, axis=-1)

    return mse_var


def mse_var_wrapper_v2(lin, labels_err):
    """
    NAME: mse_var_wrapper_v2
    PURPOSE: calculate predictive variance, and takes account of labels error
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-19 - Written - Henry Leung (University of Toronto)
    """

    def mse_var(y_true, y_pred):
        # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
        labels_err_x = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), labels_err)
        y_pred_corrected = K.log(K.exp(y_pred) + K.square(labels_err_x))
        wrapper_output = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true),
                                    0.5 * K.square(y_true - lin) * (K.exp(-y_pred_corrected)) + 0.5 * y_pred_corrected)
        return K.mean(wrapper_output, axis=-1)

    return mse_var


def mean_absolute_error(y_true, y_pred):
    """
    NAME: mean_absolute_error
    PURPOSE: calculate mean absolute error, ignoring the magic number
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    return K.mean(K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), K.abs(y_true - y_pred)),
                  axis=-1)
