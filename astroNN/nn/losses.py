# ---------------------------------------------------------------#
#   astroNN.nn.losses: losses
# ---------------------------------------------------------------#

import tensorflow as tf
from tensorflow import distributions

from astroNN.config import MAGIC_NUMBER
from astroNN.config import keras_import_manager
from astroNN.nn import magic_correction_term, nn_obj_lookup

keras = keras_import_manager()
epsilon = keras.backend.epsilon
Model = keras.models.Model


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean square error losses

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :return: Mean Squared Error
    :rtype: tf.Tensor
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    return tf.reduce_mean(tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                                   tf.square(y_true - y_pred)), axis=-1) * magic_correction_term(y_true)


def mse_lin_wrapper(var, labels_err):
    """
    Calculate predictive variance, and takes account of labels error in Bayesian Neural Network

    :param var: Predictive Variance
    :type var: Union(tf.Tensor, tf.Variable)
    :param labels_err: Known labels error, give zeros if unknown/unavailable
    :type labels_err: Union(tf.Tensor, tf.Variable)
    :return: Robust MSE function for labels prediction neurones, which matches Keras losses API
    :rtype: function
    :Returned Funtion Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Prediction
            |   Return (*tf.Tensor*): Robust Mean Squared Error
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_lin(y_true, y_pred):
        return robust_mse(y_true, y_pred, var, labels_err)

    mse_lin.__name__ = 'mse_lin_wrapper'  # set the name to be the same as parent so it can be found

    return mse_lin


def mse_var_wrapper(lin, labels_err):
    """
    Calculate predictive variance, and takes account of labels error in Bayesian Neural Network

    :param lin: Prediction
    :type lin: Union(tf.Tensor, tf.Variable)
    :param labels_err: Known labels error, give zeros if unknown/unavailable
    :type labels_err: Union(tf.Tensor, tf.Variable)
    :return: Robust MSE function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Funtion Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Predictive Variance
            |   Return (*tf.Tensor*): Robust Mean Squared Error
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_var(y_true, y_pred):
        return robust_mse(y_true, lin, y_pred, labels_err)

    mse_var.__name__ = 'mse_var_wrapper'  # set the name to be the same as parent so it can be found

    return mse_var


def robust_mse(y_true, y_pred, variance, labels_err):
    """
    Calculate predictive variance, and takes account of labels error in Bayesian Neural Network

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :param variance: Predictive Variance
    :type variance: Union(tf.Tensor, tf.Variable)
    :param labels_err: Known labels error, give zeros if unknown/unavailable
    :type labels_err: Union(tf.Tensor, tf.Variable)
    :return: Robust Mean Squared Error, can be used directly with Tensorflow
    :rtype: tf.Tensor
    :History: 2018-April-07 - Written - Henry Leung (University of Toronto)
    """
    # labels_err still contains magic_number
    labels_err_y = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), labels_err)
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    y_pred_corrected = tf.log(tf.exp(variance) + tf.square(labels_err_y))

    wrapper_output = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                              0.5 * tf.square(y_true - y_pred) * (tf.exp(-y_pred_corrected)) + 0.5 *
                              y_pred_corrected)

    return tf.reduce_mean(wrapper_output, axis=-1) * magic_correction_term(y_true)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate mean absolute error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :return: Mean Absolute Error
    :rtype: tf.Tensor
    :History: 2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    return tf.reduce_mean(tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true),
                                   tf.abs(y_true - y_pred)), axis=-1) * magic_correction_term(y_true)


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :return: Mean Absolute Percentage Error
    :rtype: tf.Tensor
    :History: 2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    tf_inf = tf.cast(tf.constant(1) / tf.constant(0), tf.float32)
    epsilon_tensor = tf.cast(tf.constant(keras.backend.epsilon()), tf.float32)

    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), epsilon_tensor, tf_inf))
    diff_corrected = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), diff)
    return 100. * tf.reduce_mean(diff_corrected, axis=-1) * magic_correction_term(y_true)


def mean_squared_logarithmic_error(y_true, y_pred):
    """
    Calculate mean squared logarithmic error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :return: Mean Squared Logarithmic Error
    :rtype: tf.Tensor
    :History: 2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    tf_inf = tf.cast(tf.constant(1) / tf.constant(0), tf.float32)
    epsilon_tensor = tf.cast(tf.constant(keras.backend.epsilon()), tf.float32)

    first_log = tf.log(tf.clip_by_value(y_pred, epsilon_tensor, tf_inf) + 1.)
    second_log = tf.log(tf.clip_by_value(y_true, epsilon_tensor, tf_inf) + 1.)
    log_diff = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), tf.square(first_log - second_log))
    return tf.reduce_mean(log_diff, axis=-1) * magic_correction_term(y_true)


def categorical_crossentropy(y_true, y_pred, from_logits=False):
    """
    Categorical cross-entropy between an output tensor and a target tensor, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :param from_logits: From logits space or not. If you want to use logits, please use from_logits=True
    :type from_logits: boolean
    :return: Categorical Cross-Entropy
    :rtype: tf.Tensor
    :History: 2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    # calculate correction term first
    correction = magic_correction_term(y_true)

    # Deal with magic number
    y_true = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), y_true)

    # Note: tf.nn.softmax_cross_entropy_with_logits_v2 expects logits, we expects probabilities by default.
    if not from_logits:
        epsilon_tensor = tf.cast(tf.constant(keras.backend.epsilon()), tf.float32)
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
        # manual computation of crossentropy
        y_pred = tf.clip_by_value(y_pred, epsilon_tensor, 1. - epsilon_tensor)
        return - tf.reduce_sum(y_true * tf.log(y_pred), len(y_pred.get_shape()) - 1) * correction
    else:
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred) * correction


def binary_crossentropy(y_true, y_pred, from_logits=False):
    """
    Binary cross-entropy between an output tensor and a target tensor, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :param from_logits: From logits space or not. If you want to use logits, please use from_logits=True
    :type from_logits: boolean
    :return: Binary Cross-Entropy
    :rtype: tf.Tensor
    :History: 2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects logits, we expects probabilities by default.
    if not from_logits:
        epsilon_tensor = tf.cast(tf.constant(keras.backend.epsilon()), tf.float32)
        # transform back to logits
        y_pred = tf.clip_by_value(y_pred, epsilon_tensor, 1. - epsilon_tensor)
        y_pred = tf.log(y_pred / (1. - y_pred))

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    corrected_cross_entropy = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(cross_entropy), cross_entropy)

    return tf.reduce_mean(corrected_cross_entropy, axis=-1) * magic_correction_term(y_true)


def bayesian_categorical_crossentropy_wrapper(logit_var):
    """
    | Categorical crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logit_var: Predictive variance
    :type logit_var: Union(tf.Tensor, tf.Variable)
    :return: Robust categorical_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Prediction in logits space
            |   Return (*tf.Tensor*): Robust categorical crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is logits
    def bayesian_crossentropy(y_true, y_pred):
        return robust_categorical_crossentropy(y_true, y_pred, logit_var)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = 'bayesian_categorical_crossentropy_wrapper'

    return bayesian_crossentropy


def bayesian_categorical_crossentropy_var_wrapper(logits):
    """
    | Categorical crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logits: Prediction in logits space
    :type logits: Union(tf.Tensor, tf.Variable)
    :return: Robust categorical_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Predictive variance in logits space
            |   Return (*tf.Tensor*): Robust categorical crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is predictive entropy
    def bayesian_crossentropy(y_true, y_pred):
        return robust_categorical_crossentropy(y_true, logits, y_pred)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = 'bayesian_categorical_crossentropy_var_wrapper'

    return bayesian_crossentropy


def robust_categorical_crossentropy(y_true, y_pred, logit_var):
    """
    Calculate categorical accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction in logits space
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :param logit_var: Predictive variance in logits space
    :type logit_var: Union(tf.Tensor, tf.Variable)
    :return: categorical cross-entropy
    :rtype: tf.Tensor
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """
    variance_depressor = tf.reduce_mean(tf.exp(logit_var) - tf.ones_like(logit_var))
    undistorted_loss = categorical_crossentropy(y_true, y_pred, from_logits=True)
    dist = distributions.Normal(loc=y_pred, scale=logit_var)

    mc_result = tf.map_fn(
        lambda x: -tf.nn.elu(undistorted_loss - categorical_crossentropy(y_true, x, from_logits=True)),
        dist.sample([25]), dtype=tf.float32)

    variance_loss = tf.reduce_mean(mc_result, axis=0) * undistorted_loss

    return (variance_loss + undistorted_loss + variance_depressor) * magic_correction_term(y_true)


def bayesian_binary_crossentropy_wrapper(logit_var, mc_num):
    """
    | Binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logit_var: Predictive variance
    :type logit_var: Union(tf.Tensor, tf.Variable)
    :return: Robust binary_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Prediction in logits space
            |   Return (*tf.Tensor*): Robust binary crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """
    # y_pred is logits
    def bayesian_crossentropy(y_true, y_pred):
        return robust_binary_crossentropy(y_true, y_pred, logit_var)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = 'bayesian_binary_crossentropy_wrapper'

    return bayesian_crossentropy


def bayesian_binary_crossentropy_var_wrapper(logits, mc_num):
    """
    | Binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logits: Prediction in logits space
    :type logits: Union(tf.Tensor, tf.Variable)
    :return: Robust binary_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Predictive variance in logits space
            |   Return (*tf.Tensor*): Robust binary crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is predictive entropy
    def bayesian_crossentropy(y_true, y_pred):
        return robust_binary_crossentropy(y_true, logits, y_pred)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = 'bayesian_binary_crossentropy_var_wrapper'

    return bayesian_crossentropy


def robust_binary_crossentropy(y_true, y_pred, logit_var):
    """
    Calculate binary accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction in logits space
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :param logit_var: Predictive variance in logits space
    :type logit_var: Union(tf.Tensor, tf.Variable)
    :return: categorical cross-entropy
    :rtype: tf.Tensor
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """
    variance_depressor = tf.reduce_mean(tf.exp(logit_var) - tf.ones_like(logit_var))
    undistorted_loss = binary_crossentropy(y_true, y_pred, from_logits=True)
    dist = distributions.Normal(loc=y_pred, scale=logit_var)

    mc_result = tf.map_fn(
        lambda x: -tf.nn.elu(undistorted_loss - binary_crossentropy(y_true, x, from_logits=True)),
        dist.sample([25]), dtype=tf.float32)

    variance_loss = tf.reduce_mean(mc_result, axis=0) * undistorted_loss

    return (variance_loss + undistorted_loss + variance_depressor) * magic_correction_term(y_true)


def nll(y_true, y_pred):
    """
    NAME: nll
    PURPOSE:
        Negative log likelihood
    INPUT:
        No input for users
    OUTPUT:
        (tf.Tensor)
    HISTORY:
        2018-Jan-30 - Written - Henry Leung (University of Toronto)
    """
    # astroNN binary_cross_entropy gives the mean over the last axis. we require the sum
    return tf.reduce_sum(binary_crossentropy(y_true, y_pred), axis=-1)


def categorical_accuracy(y_true, y_pred):
    """
    Calculate categorical accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(tf.Tensor, tf.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(tf.Tensor, tf.Variable)
    :return: Categorical Classification Accuracy
    :rtype: tf.Tensor
    :History: 2018-Jan-21 - Written - Henry Leung (University of Toronto)
    """
    y_true = tf.where(tf.equal(y_true, MAGIC_NUMBER), tf.zeros_like(y_true), y_true)
    return tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)),
                   tf.float32) * magic_correction_term(y_true)


def binary_accuracy(from_logits=False):
    """
    Calculate binary accuracy, ignoring the magic number

    :param from_logits: From logits space or not. If you want to use logits, please use from_logits=True
    :type from_logits: boolean
    :return: Function for Binary classification accuracy which matches Keras losses API
    :rtype: function
    :Returned Funtion Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*tf.Tensor*): Ground Truth
            |   - **y_pred** (*tf.Tensor*): Prediction
            |   Return (*tf.Tensor*): Binary Classification Accuracy
    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """

    # DO NOT correct y_true for magic number, just let it goes wrong and then times a correction terms
    def binary_accuracy_internal(y_true, y_pred):
        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32), axis=-1) * magic_correction_term(
            y_true)

    binary_accuracy_internal.__name__ = 'binary_accuracy'  # set the name to be displayed in TF/Keras log

    return binary_accuracy_internal


# Just alias functions
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
msle = mean_squared_logarithmic_error

# legacy suppert
mse_lin = mse_lin_wrapper
mse_var = mse_var_wrapper


def losses_lookup(identifier):
    """
    Lookup astroNN.nn.losses function by name

    :param identifier: identifier
    :type identifier: str
    :return: Looked up function
    :rtype: function
    :History: 2018-Apr-28 - Written - Henry Leung (University of Toronto)
    """
    return nn_obj_lookup(identifier, module_obj=globals(), module_name=__name__)
