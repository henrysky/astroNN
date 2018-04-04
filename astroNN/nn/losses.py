import tensorflow as tf

from astroNN.config import MAGIC_NUMBER
from astroNN.config import keras_import_manager
from astroNN.nn import magic_correction_term

keras = keras_import_manager()
epsilon = keras.backend.epsilon
Model = keras.models.Model

tf_inf = tf.cast(tf.constant(1) / tf.constant(0), tf.float32)  # has a better performance than keras convert to tensor
epsilon_tensor = tf.cast(tf.constant(keras.backend.epsilon()), tf.float32)


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
    first_log = tf.log(tf.clip_by_value(y_pred, epsilon_tensor, tf_inf) + 1.)
    second_log = tf.log(tf.clip_by_value(y_true, epsilon_tensor, tf_inf) + 1.)
    log_diff = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), tf.square(first_log - second_log))
    return tf.reduce_mean(log_diff, axis=-1) * magic_correction_term(y_true)


def categorical_cross_entropy(y_true, y_pred, from_logits=False):
    """
    NAME: astronn_categorical_crossentropy
    PURPOSE: Categorical crossentropy between an output tensor and a target tensor.
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
    # calculate correction term first
    correction = magic_correction_term(y_true)

    # Deal with magic number
    y_true = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), y_true)

    # Note: tf.nn.softmax_cross_entropy_with_logits_v2 expects logits, we expects probabilities by default.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
        # manual computation of crossentropy
        y_pred = tf.clip_by_value(y_pred, epsilon_tensor, 1. - epsilon_tensor)
        return - tf.reduce_sum(y_true * tf.log(y_pred), len(y_pred.get_shape()) - 1) * correction
    else:
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred) * correction


def binary_cross_entropy(y_true, y_pred, from_logits=False):
    """
    NAME: binary_crossentropy
    PURPOSE: Binary crossentropy between an output tensor and a target tensor.
    INPUT:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a sigmoid (unless `from_logits` is True, in which case `output` is expected
        to be the logits).
        from_logits: Boolean, whether `output` is the result of a sigmoid, or is a tensor of logits.
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """

    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects logits, we expects probabilities by default.
    if not from_logits:
        # transform back to logits
        y_pred = tf.clip_by_value(y_pred, epsilon_tensor, 1 - epsilon_tensor)
        y_pred = tf.log(y_pred / (1 - y_pred))

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    corrected_cross_entropy = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(cross_entropy), cross_entropy)

    return tf.reduce_mean(corrected_cross_entropy, axis=-1) * magic_correction_term(y_true)


def bayesian_categorical_crossentropy_wrapper(logit_var, mc_num):
    """
    NAME: bayesian_categorical_crossentropy_wrapper
    PURPOSE: categorical crossentropy between an output tensor and a target tensor for Bayesian Neural Network
            equation (12) of arxiv:1703.04977
    INPUT:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected
        to be the logits).
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is logits
    def bayesian_crossentropy(y_true, y_pred):
        variance_depressor = tf.reduce_mean(tf.exp(logit_var) - tf.ones_like(logit_var))
        undistorted_loss = categorical_cross_entropy(y_true, y_pred, from_logits=True)
        dist = tf.distributions.Normal(loc=y_pred, scale=tf.sqrt(logit_var))
        mc_result = - tf.map_fn(gaussian_categorical_crossentropy(y_true, dist, undistorted_loss), tf.ones(mc_num))
        variance_loss = tf.reduce_mean(mc_result, axis=0) * undistorted_loss
        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_crossentropy


def bayesian_categorical_crossentropy_var_wrapper(logits, mc_num):
    """
    NAME: bayesian_categorical_crossentropy_var_wrapper
    PURPOSE: categorical crossentropy between an output tensor and a target tensor for Bayesian Neural Network
            equation (12) of arxiv:1703.04977
    INPUT:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax (unless `from_logits` is True, in which case `output` is expected
        to be the logits).
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is predictive entropy
    def bayesian_crossentropy(y_true, y_pred):
        variance_depressor = tf.reduce_mean(tf.exp(y_pred) - tf.ones_like(y_pred))
        undistorted_loss = categorical_cross_entropy(y_true, logits, from_logits=True)
        dist = tf.distributions.Normal(loc=logits, scale=tf.sqrt(y_pred))
        mc_result = - tf.map_fn(gaussian_categorical_crossentropy(y_true, dist, undistorted_loss), tf.ones(mc_num))
        variance_loss = tf.reduce_mean(mc_result, axis=0) * undistorted_loss
        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_crossentropy


def gaussian_categorical_crossentropy(true, dist, undistorted_loss):
    """
    NAME: gaussian_categorical_crossentropy
    PURPOSE: used for corrupting the logits
    INPUT:
        You should not ue this directly
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-15 - Written - Henry Leung (University of Toronto)
        Credit: https://github.com/kyle-dorman/bayesian-neural-network-blogpost
    """

    def map_fn(i):
        distorted_loss = categorical_cross_entropy(true, dist.sample([1]), from_logits=True)
        return tf.nn.elu(undistorted_loss - distorted_loss)

    return map_fn


def bayesian_binary_crossentropy_wrapper(logit_var, mc_num):
    """
    NAME: bayesian_binary_crossentropy_wrapper
    PURPOSE: binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
            equation (12) of arxiv:1703.04977
    INPUT:
        logit_var: A tensor of the same shape as `output`.
        mc_num: number of monte carlo run
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is logits
    def bayesian_crossentropy(y_true, y_pred):
        variance_depressor = tf.reduce_mean(tf.exp(y_pred) - tf.ones_like(y_pred))
        undistorted_loss = binary_cross_entropy(y_true, y_pred, from_logits=True)
        dist = tf.distributions.Normal(loc=y_pred, scale=tf.sqrt(logit_var))
        mc_result = - tf.map_fn(gaussian_binary_crossentropy(y_true, dist, undistorted_loss), tf.ones(mc_num))
        variance_loss = tf.reduce_mean(mc_result, axis=0) * undistorted_loss
        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_crossentropy


def bayesian_binary_crossentropy_var_wrapper(logits, mc_num):
    """
    NAME: bayesian_binary_crossentropy_var_wrapper
    PURPOSE: Binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
            equation (12) of arxiv:1703.04977
    INPUT:
        logits: A tensor of the same shape as `output`.
        mc_num: number of monte carlo run
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is predictive logits variance
    def bayesian_crossentropy(y_true, y_pred):
        variance_depressor = tf.reduce_mean(tf.exp(y_pred) - tf.ones_like(y_pred))
        undistorted_loss = binary_cross_entropy(y_true, logits, from_logits=True)
        dist = tf.distributions.Normal(loc=logits, scale=tf.sqrt(y_pred))
        mc_result = - tf.map_fn(gaussian_binary_crossentropy(y_true, dist, undistorted_loss), tf.ones(mc_num))
        variance_loss = tf.reduce_mean(mc_result, axis=0) * undistorted_loss
        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_crossentropy


def gaussian_binary_crossentropy(true, dist, undistorted_loss):
    """
    NAME: gaussian_binary_crossentropy
    PURPOSE: used for corrupting the logits
    INPUT:
        You should not ue this directly
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-15 - Written - Henry Leung (University of Toronto)
        Credit: https://github.com/kyle-dorman/bayesian-neural-network-blogpost
    """

    def map_fn(i):
        # need to expand due to a weird shape issue
        distorted_loss = binary_cross_entropy(tf.expand_dims(true, axis=0), dist.sample(1), from_logits=True)
        return tf.nn.elu(undistorted_loss - distorted_loss)

    return map_fn


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
