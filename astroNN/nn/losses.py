import tensorflow as tf

from astroNN import MAGIC_NUMBER
from astroNN.nn import magic_correction_term
from astroNN import keras_import_manager

keras = keras_import_manager()
epsilon = keras.backend.epsilon
Model = keras.models.Model

tf_inf = tf.cast(tf.constant(1) / tf.constant(0), tf.float32)


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
                                   tf.square(y_true - y_pred)), axis=-1) * magic_correction_term(y_true)


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

        return tf.reduce_mean(wrapper_output, axis=-1) * magic_correction_term(y_true)

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

        return tf.reduce_mean(wrapper_output, axis=-1) * magic_correction_term(y_true)

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


def mean_absolute_percentage_error(y_true, y_pred):
    """
    NAME: mean_absolute_percentage_error
    PURPOSE: calculate mean absolute percentage error, ignoring the magic number
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    epsilon_tensor = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), epsilon_tensor, tf_inf))
    diff_corrected = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), diff)
    return 100. * tf.reduce_mean(diff_corrected, axis=-1) * magic_correction_term(y_true)


def mean_squared_logarithmic_error(y_true, y_pred):
    """
    NAME: mean_squared_logarithmic_error
    PURPOSE: calculate mean squared logarithmic error, ignoring the magic number
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    epsilon_tensor = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    first_log = tf.log(tf.clip_by_value(y_pred, epsilon_tensor, tf_inf) + 1.)
    second_log = tf.log(tf.clip_by_value(y_true, epsilon_tensor, tf_inf) + 1.)
    msle = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), tf.square(first_log - second_log))
    return tf.reduce_mean(msle, axis=-1) * magic_correction_term(y_true)


def categorical_cross_entropy(y_true, y_pred, from_logits=False):
    """
    NAME: astronn_categorical_crossentropy
    PURPOSE: Categorical crossentropy between an output tensor and a target tensor.
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
    INPUT:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected
        to be the logits).
        from_logits: Boolean, whether `output` is the result of a softmax, or is a tensor of logits.
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    # Deal with magic number first
    y_true = tf.where(tf.equal(y_true, MAGIC_NUMBER), y_pred, y_true)

    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
        # manual computation of crossentropy
        epsilon_tensor = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon_tensor, 1. - epsilon_tensor)
        return - tf.reduce_sum(y_true * tf.log(y_pred), len(y_pred.get_shape()) - 1) * magic_correction_term(y_true)
    else:
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred) * \
               magic_correction_term(y_true)


def binary_cross_entropy(y_true, y_pred, from_logits=False):
    """
    NAME: binary_crossentropy
    PURPOSE: Binary crossentropy between an output tensor and a target tensor.
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
    INPUT:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected
        to be the logits).
        from_logits: Boolean, whether `output` is the result of a softmax, or is a tensor of logits.
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    # Deal with magic number first
    y_true = tf.where(tf.equal(y_true, MAGIC_NUMBER), y_pred, y_true)

    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        epsilon_tensor = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon_tensor, 1 - epsilon_tensor)
        y_pred = tf.log(y_pred / (1 - y_pred))

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1) * \
           magic_correction_term(y_true)


def bayesian_crossentropy_wrapper():
    """
    NAME: bayesian_crossentropy_wrapper
    PURPOSE: Binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
    INPUT:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected
        to be the logits).
        from_logits: Boolean, whether `output` is the result of a softmax, or is a tensor of logits.
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-01 - Written - Henry Leung (University of Toronto)
    """

    def bayesian_crossentropy(y_true, y_pred):
        mc_log_softmax = y_pred - tf.maximum(y_pred, axis=2, keepdims=True)
        mc_log_softmax = mc_log_softmax - tf.log(tf.reduce_sum(tf.exp(mc_log_softmax), axis=2, keepdims=True))
        mc_ll = tf.reduce_sum(y_true * mc_log_softmax, -1)  # N x K
        # K_mc = mc_ll.get_shape().as_list()[1]	# only for tensorflow
        return - (tf.reduce_logsumexp(mc_ll, 1) + tf.log(1.0 / mc_ll))

    return bayesian_crossentropy


def nll(y_true, y_pred):
    """
    NAME: nll
    PURPOSE:
        Negative log likelihood
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-30 - Written - Henry Leung (University of Toronto)
    """
    # astroNN binary_cross_entropy gives the mean over the last axis. we require the sum
    return tf.reduce_sum(binary_cross_entropy(y_true, y_pred), axis=-1)


# Just alias functions
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
