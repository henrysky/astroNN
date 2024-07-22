# ---------------------------------------------------------------#
#   astroNN.nn.losses: losses
# ---------------------------------------------------------------#

import keras

from astroNN.config import MAGIC_NUMBER
from astroNN.nn import nn_obj_lookup

epsilon = keras.backend.epsilon
Model = keras.Model


def magic_num_check(x):
    # check for magic num and nan
    return keras.ops.logical_or(
        keras.ops.equal(x, MAGIC_NUMBER), keras.ops.isnan(x)
    )


def magic_correction_term(y_true):
    """
    Calculate a correction term to prevent the loss being "lowered" by magic_num or NaN

    :param y_true: Ground Truth
    :type y_true: keras.ops.Tensor
    :return: Correction Term
    :rtype: keras.ops.Tensor
    :History:
        | 2018-Jan-30 - Written - Henry Leung (University of Toronto)
        | 2018-Feb-17 - Updated - Henry Leung (University of Toronto)
    """

    num_nonmagic = keras.ops.sum(
        keras.ops.cast(
            keras.ops.logical_not(magic_num_check(y_true)),
            "float32",
        ),
        axis=-1,
    )
    num_magic = keras.ops.sum(
        keras.ops.cast(magic_num_check(y_true), "float32"), axis=-1
    )

    # If no magic number, then num_zero=0 and whole expression is just 1 and get back our good old loss
    # If num_nonzero is 0, that means we don't have any information, then set the correction term to ones
    return (num_nonmagic + num_magic) / num_nonmagic


def weighted_loss(losses, sample_weight=None):
    """
    Calculate sample-weighted losses from losses

    :param losses: Losses
    :type losses: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Weighted loss
    :rtype: keras.ops.Tensor
    :History: 2021-Feb-02 - Written - Henry Leung (University of Toronto)
    """
    if sample_weight is None:
        return losses
    else:
        return keras.ops.multiply(losses, sample_weight)


def median(x, axis=None):
    """
    Calculate median

    :param x: Data
    :type x: keras.ops.Tensor
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: keras.ops.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """

    def median_internal(_x):
        shape = keras.ops.shape(_x)[0]
        if shape % 2 == 1:
            _median = keras.ops.top_k(_x, shape // 2 + 1).values[-1]
        else:
            _median = (
                keras.ops.top_k(_x, shape // 2).values[-1]
                + keras.ops.top_k(_x, shape // 2 + 1).values[-1]
            ) / 2
        return _median

    if axis is None:
        x_flattened = keras.ops.reshape(x, [-1])
        median = median_internal(x_flattened)
        return median
    else:
        x_unstacked = keras.ops.unstack(
            keras.ops.transpose(x), axis=axis
        )
        median = keras.ops.stack([median_internal(_x) for _x in x_unstacked])
        return median


def mean_squared_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean square error losses

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Mean Squared Error
    :rtype: keras.ops.Tensor
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """
    losses = keras.ops.mean(
        keras.ops.where(
            magic_num_check(y_true),
            keras.ops.zeros_like(y_true),
            keras.ops.square(y_true - y_pred),
        ),
        axis=-1,
    ) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def mean_squared_reconstruction_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean square reconstruction error losses

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Mean Squared Error
    :rtype: keras.ops.Tensor
    :History: 2022-May-05 - Written - Henry Leung (University of Toronto)
    """
    raw_loss = keras.ops.square(y_true - y_pred)
    fixed_loss = keras.ops.where(
        magic_num_check(y_true),
        keras.ops.zeros_like(y_true),
        raw_loss,
    )
    losses = weighted_loss(keras.ops.mean(fixed_loss, axis=-1), sample_weight)
    return keras.ops.mean(keras.ops.sum(losses, axis=-1))


def mse_lin_wrapper(var, labels_err):
    """
    Calculate predictive variance, and takes account of labels error in Bayesian Neural Network

    :param var: Predictive Variance
    :type var: Union(keras.ops.Tensor, keras.ops.Variable)
    :param labels_err: Known labels error, give zeros if unknown/unavailable
    :type labels_err: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Robust MSE function for labels prediction neurones, which matches Keras losses API
    :rtype: function
    :Returned Funtion Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Prediction
            |   Return (*keras.ops.Tensor*): Robust Mean Squared Error
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_lin(y_true, y_pred, sample_weight=None):
        return robust_mse(y_true, y_pred, var, labels_err, sample_weight)

    mse_lin.__name__ = (
        "mse_lin_wrapper"  # set the name to be the same as parent so it can be found
    )

    return mse_lin


def mse_var_wrapper(lin, labels_err):
    """
    Calculate predictive variance, and takes account of labels error in Bayesian Neural Network

    :param lin: Prediction
    :type lin: Union(keras.ops.Tensor, keras.ops.Variable)
    :param labels_err: Known labels error, give zeros if unknown/unavailable
    :type labels_err: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Robust MSE function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Funtion Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Predictive Variance
            |   Return (*keras.ops.Tensor*): Robust Mean Squared Error
    :History: 2017-Nov-16 - Written - Henry Leung (University of Toronto)
    """

    def mse_var(y_true, y_pred, sample_weight=None):
        return robust_mse(y_true, lin, y_pred, labels_err, sample_weight)

    mse_var.__name__ = (
        "mse_var_wrapper"  # set the name to be the same as parent so it can be found
    )

    return mse_var


def robust_mse(y_true, y_pred, variance, labels_err, sample_weight=None):
    """
    Calculate predictive variance, and takes account of labels error in Bayesian Neural Network

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param variance: Log Predictive Variance
    :type variance: Union(keras.ops.Tensor, keras.ops.Variable)
    :param labels_err: Known labels error, give zeros if unknown/unavailable
    :type labels_err: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Robust Mean Squared Error 
    :rtype: keras.ops.Tensor
    :History: 2018-April-07 - Written - Henry Leung (University of Toronto)
    """
    # labels_err still contains magic_number
    labels_err = keras.ops.cast(labels_err, "float32")
    labels_err_y = keras.ops.where(
        magic_num_check(y_true),
        keras.ops.zeros_like(y_true),
        labels_err,
    )
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    y_pred_corrected = keras.ops.log(
        keras.ops.exp(variance) + keras.ops.square(labels_err_y)
    )

    raw_output = 0.5 * keras.ops.square(y_true - y_pred) * (keras.ops.exp(-y_pred_corrected)) + 0.5 * y_pred_corrected
    wrapper_output = keras.ops.where(
        magic_num_check(y_true),
        keras.ops.zeros_like(y_true),
        raw_output
    )

    losses = keras.ops.mean(wrapper_output, axis=-1) * magic_correction_term(
        y_true
    )
    return weighted_loss(losses, sample_weight)


def mean_absolute_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean absolute error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Mean Absolute Error
    :rtype: keras.ops.Tensor
    :History: 2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    raw_loss = keras.ops.abs(y_true - y_pred)
    losses = keras.ops.mean(
        keras.ops.where(
            magic_num_check(y_true),
            keras.ops.zeros_like(y_true),
            raw_loss,
        ),
        axis=-1,
    ) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean absolute percentage error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Mean Absolute Percentage Error
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :rtype: keras.ops.Tensor
    :History: 2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    keras.ops_inf = keras.ops.cast(
        keras.ops.array(1) / keras.ops.array(0),
        "float32",
    )
    epsilon_tensor = keras.ops.cast(
        keras.ops.array(keras.backend.epsilon()),
        "float32",
    )

    diff = keras.ops.abs(
        (y_true - y_pred)
        / keras.ops.clip(
            keras.ops.abs(y_true), epsilon_tensor, keras.ops_inf
        )
    )
    diff_corrected = keras.ops.where(
        magic_num_check(y_true), keras.ops.zeros_like(y_true), diff
    )
    losses = (
        100.0
        * keras.ops.mean(diff_corrected, axis=-1)
        * magic_correction_term(y_true)
    )
    return weighted_loss(losses, sample_weight)


def median_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """
    Calculate median absolute percentage error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Median Absolute Percentage Error
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :rtype: keras.ops.Tensor
    :History: 2020-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    keras.ops_inf = keras.ops.cast(
        keras.ops.array(1) / keras.ops.array(0),
        "float32",
    )
    epsilon_tensor = keras.ops.cast(
        keras.ops.array(keras.backend.epsilon()),
        "float32",
    )

    diff = keras.ops.abs(
        (y_true - y_pred)
        / keras.ops.clip(
            keras.ops.abs(y_true), epsilon_tensor, keras.ops_inf
        )
    )
    diff_corrected = keras.ops.where(
        magic_num_check(y_true), keras.ops.zeros_like(y_true), diff
    )
    losses = 100.0 * median(diff_corrected, axis=None) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def mean_squared_logarithmic_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean squared logarithmic error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Mean Squared Logarithmic Error
    :rtype: keras.ops.Tensor
    :History: 2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    keras.ops_inf = keras.ops.cast(
        keras.ops.array(1) / keras.ops.array(0),
        "float32",
    )
    epsilon_tensor = keras.ops.cast(
        keras.ops.array(keras.backend.epsilon()),
        "float32",
    )

    first_log = keras.ops.log(
        keras.ops.clip(y_pred, epsilon_tensor, keras.ops_inf) + 1.0
    )
    second_log = keras.ops.log(
        keras.ops.clip(y_true, epsilon_tensor, keras.ops_inf) + 1.0
    )
    raw_log_diff = keras.ops.square(first_log - second_log)
    log_diff = keras.ops.where(
        magic_num_check(y_true),
        keras.ops.zeros_like(y_true),
        raw_log_diff,
    )
    losses = keras.ops.mean(log_diff, axis=-1) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def mean_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean error as a way to get the bias in prediction, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Mean Error
    :rtype: keras.ops.Tensor
    :History: 2018-May-22 - Written - Henry Leung (University of Toronto)
    """
    raw_loss = y_true - y_pred
    losses = keras.ops.mean(
        keras.ops.where(
            magic_num_check(y_true),
            keras.ops.zeros_like(y_true),
            raw_loss,
        ),
        axis=-1,
    ) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def mean_percentage_error(y_true, y_pred, sample_weight=None):
    """
    Calculate mean percentage error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Mean Percentage Error
    :rtype: keras.ops.Tensor
    :History: 2018-Jun-06 - Written - Henry Leung (University of Toronto)
    """
    keras.ops_inf = keras.ops.cast(
        keras.ops.array(1) / keras.ops.array(0),
        "float32",
    )
    epsilon_tensor = keras.ops.cast(
        keras.ops.array(keras.backend.epsilon()),
        "float32",
    )

    diff = y_true - y_pred / keras.ops.clip(
        y_true, epsilon_tensor, keras.ops_inf
    )
    diff_corrected = keras.ops.where(
        magic_num_check(y_true), keras.ops.zeros_like(y_true), diff
    )
    losses = (
        100.0
        * keras.ops.mean(diff_corrected, axis=-1)
        * magic_correction_term(y_true)
    )
    return weighted_loss(losses, sample_weight)


def median_percentage_error(y_true, y_pred, sample_weight=None):
    """
    Calculate median percentage error, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Median Percentage Error
    :rtype: keras.ops.Tensor
    :History: 2020-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    keras.ops_inf = keras.ops.cast(
        keras.ops.array(1) / keras.ops.array(0),
        "float32",
    )
    epsilon_tensor = keras.ops.cast(
        keras.ops.array(keras.backend.epsilon()),
        "float32",
    )

    diff = y_true - y_pred / keras.ops.clip(
        y_true, epsilon_tensor, keras.ops_inf
    )
    diff_corrected = keras.ops.where(
        magic_num_check(y_true), keras.ops.zeros_like(y_true), diff
    )
    losses = 100.0 * median(diff_corrected, axis=None) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def categorical_crossentropy(y_true, y_pred, sample_weight=None, from_logits=False):
    """
    Categorical cross-entropy between an output tensor and a target tensor, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :param from_logits: From logits space or not. If you want to use logits, please use from_logits=True
    :type from_logits: boolean
    :return: Categorical Cross-Entropy
    :rtype: keras.ops.Tensor
    :History: 2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    # calculate correction term first
    correction = magic_correction_term(y_true)

    # Deal with magic number
    y_true = keras.ops.where(
        magic_num_check(y_true), keras.ops.zeros_like(y_true), y_true
    )

    # Note: keras.ops.softmax_cross_entropy_with_logits expects logits, we expects probabilities by default.
    if not from_logits:
        epsilon_tensor = keras.ops.cast(
            keras.ops.array(keras.backend.epsilon()),
            "float32",
        )
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / keras.ops.sum(y_pred, len(keras.ops.shape(y_pred)) - 1, True)
        # manual computation of crossentropy
        y_pred = keras.ops.clip(y_pred, epsilon_tensor, 1.0 - epsilon_tensor)
        losses = (
            -keras.ops.sum(
                y_true * keras.ops.log(y_pred), len(keras.ops.shape(y_pred)) - 1
            )
            * correction
        )
        return weighted_loss(losses, sample_weight)
    else:
        losses = (
            keras.ops.categorical_crossentropy(target=y_true, output=y_pred, from_logits=True)
            * correction
        )
        return weighted_loss(losses, sample_weight)


def binary_crossentropy(y_true, y_pred, sample_weight=None, from_logits=False):
    """
    Binary cross-entropy between an output tensor and a target tensor, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param from_logits: From logits space or not. If you want to use logits, please use from_logits=True
    :type from_logits: boolean
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Binary Cross-Entropy
    :rtype: keras.ops.Tensor
    :History: 2018-Jan-14 - Written - Henry Leung (University of Toronto)
    """
    # Note: keras.ops.sigmoid_cross_entropy_with_logits expects logits, we expects probabilities by default.
    if not from_logits:
        epsilon_tensor = keras.ops.cast(
            keras.ops.array(keras.backend.epsilon()),
            "float32",
        )
        # transform back to logits
        y_pred = keras.ops.clip(y_pred, epsilon_tensor, 1.0 - epsilon_tensor)
        y_pred = keras.ops.log(y_pred / (1.0 - y_pred))

    cross_entropy = keras.ops.binary_crossentropy(
        target=y_true, output=y_pred, from_logits=True
    )
    corrected_cross_entropy = keras.ops.where(
        magic_num_check(y_true),
        keras.ops.zeros_like(cross_entropy),
        cross_entropy,
    )

    losses = keras.ops.mean(
        corrected_cross_entropy, axis=-1
    ) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def bayesian_categorical_crossentropy_wrapper(logit_var, *args, **kwargs):
    """
    | Categorical crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logit_var: Predictive variance
    :type logit_var: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Robust categorical_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Prediction in logits space
            |   Return (*keras.ops.Tensor*): Robust categorical crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is logits
    def bayesian_crossentropy(y_true, y_pred, sample_weight=None):
        return robust_categorical_crossentropy(y_true, y_pred, logit_var, sample_weight)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = "bayesian_categorical_crossentropy_wrapper"

    return bayesian_crossentropy


def bayesian_categorical_crossentropy_var_wrapper(logits):
    """
    | Categorical crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logits: Prediction in logits space
    :type logits: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Robust categorical_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Predictive variance in logits space
            |   Return (*keras.ops.Tensor*): Robust categorical crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is predictive entropy
    def bayesian_crossentropy(y_true, y_pred, sample_weight=None):
        return robust_categorical_crossentropy(y_true, logits, y_pred, sample_weight)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = "bayesian_categorical_crossentropy_var_wrapper"

    return bayesian_crossentropy


def robust_categorical_crossentropy(y_true, y_pred, logit_var, sample_weight):
    """
    Calculate categorical accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction in logits space
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param logit_var: Predictive variance in logits space
    :type logit_var: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: categorical cross-entropy
    :rtype: keras.ops.Tensor
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """
    variance_depressor = keras.ops.mean(
        keras.ops.exp(logit_var) - keras.ops.ones_like(logit_var)
    )
    undistorted_loss = categorical_crossentropy(
        y_true, y_pred, sample_weight, from_logits=True
    )
    mc_num = 25
    batch_size = keras.ops.shape(y_pred)[0]
    label_size = keras.ops.shape(y_pred)[-1]
    noise = keras.random.normal(shape=[mc_num, batch_size, label_size])
    dist = y_pred + noise * logit_var
    mc_result = -keras.ops.elu(
        keras.ops.tile(undistorted_loss, [mc_num])
        - categorical_crossentropy(
            keras.ops.tile(y_true, [mc_num, 1]),
            keras.ops.reshape(dist, (batch_size * mc_num, label_size)),
            from_logits=True,
        )
    )

    variance_loss = (
        keras.ops.mean(
            keras.ops.reshape(mc_result, (mc_num, batch_size)), axis=0
        )
        * undistorted_loss
    )

    losses = (
        variance_loss + undistorted_loss + variance_depressor
    ) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def bayesian_binary_crossentropy_wrapper(logit_var, *args, **kwargs):
    """
    | Binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logit_var: Predictive variance
    :type logit_var: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Robust binary_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Prediction in logits space
            |   Return (*keras.ops.Tensor*): Robust binary crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is logits
    def bayesian_crossentropy(y_true, y_pred, sample_weight=None):
        return robust_binary_crossentropy(y_true, y_pred, logit_var, sample_weight)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = "bayesian_binary_crossentropy_wrapper"

    return bayesian_crossentropy


def bayesian_binary_crossentropy_var_wrapper(logits):
    """
    | Binary crossentropy between an output tensor and a target tensor for Bayesian Neural Network
    | equation (12) of arxiv:1703.04977

    :param logits: Prediction in logits space
    :type logits: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Robust binary_crossentropy function for predictive variance neurones which matches Keras losses API
    :rtype: function
    :Returned Function Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Predictive variance in logits space
            |   Return (*keras.ops.Tensor*): Robust binary crossentropy
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """

    # y_pred is predictive entropy
    def bayesian_crossentropy(y_true, y_pred, sample_weight=None):
        return robust_binary_crossentropy(y_true, logits, y_pred, sample_weight)

    # set the name to be the same as parent so it can be found
    bayesian_crossentropy.__name__ = "bayesian_binary_crossentropy_var_wrapper"

    return bayesian_crossentropy


def robust_binary_crossentropy(y_true, y_pred, logit_var, sample_weight):
    """
    Calculate binary accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction in logits space
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param logit_var: Predictive variance in logits space
    :type logit_var: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: categorical cross-entropy
    :rtype: keras.ops.Tensor
    :History: 2018-Mar-15 - Written - Henry Leung (University of Toronto)
    """
    variance_depressor = keras.ops.mean(
        keras.ops.exp(logit_var) - keras.ops.ones_like(logit_var)
    )
    undistorted_loss = binary_crossentropy(y_true, y_pred, from_logits=True)
    mc_num = 25
    batch_size = keras.ops.shape(y_pred)[0]
    label_size = keras.ops.shape(y_pred)[-1]
    noise = keras.random.normal(shape=[mc_num, batch_size, label_size])
    dist = y_pred + noise * logit_var
    mc_result = -keras.ops.elu(
        keras.ops.tile(undistorted_loss, [mc_num])
        - binary_crossentropy(
            keras.ops.tile(y_true, [mc_num, 1]),
            keras.ops.reshape(dist, (batch_size * mc_num, label_size)),
            from_logits=True,
        )
    )

    variance_loss = (
        keras.ops.mean(
            keras.ops.reshape(mc_result, (mc_num, batch_size)), axis=0
        )
        * undistorted_loss
    )

    losses = (
        variance_loss + undistorted_loss + variance_depressor
    ) * magic_correction_term(y_true)
    return weighted_loss(losses, sample_weight)


def nll(y_true, y_pred, sample_weight=None):
    """
    Calculate negative log likelihood

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Negative log likelihood
    :rtype: keras.ops.Tensor
    :History: 2018-Jan-30 - Written - Henry Leung (University of Toronto)
    """
    # astroNN binary_cross_entropy gives the mean over the last axis. we require the sum
    losses = keras.ops.sum(binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss(losses, sample_weight)


def categorical_accuracy(y_true, y_pred):
    """
    Calculate categorical accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Categorical Classification Accuracy
    :rtype: keras.ops.Tensor
    :History: 2018-Jan-21 - Written - Henry Leung (University of Toronto)
    """
    y_true = keras.ops.where(
        magic_num_check(y_true), keras.ops.zeros_like(y_true), y_true
    )
    return keras.ops.cast(
        keras.ops.equal(
            keras.ops.argmax(y_true, axis=-1),
            keras.ops.argmax(y_pred, axis=-1),
        ),
        "float32",
    ) * magic_correction_term(y_true)


def __binary_accuracy(from_logits=False):
    """
    Calculate binary accuracy, ignoring the magic number

    :param from_logits: From logits space or not. If you want to use logits, please use from_logits=True
    :type from_logits: boolean
    :return: Function for Binary classification accuracy which matches Keras losses API
    :rtype: function
    :Returned Funtion Parameter:
            | **function(y_true, y_pred)**
            |   - **y_true** (*keras.ops.Tensor*): Ground Truth
            |   - **y_pred** (*keras.ops.Tensor*): Prediction
            |   Return (*keras.ops.Tensor*): Binary Classification Accuracy
    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """

    # DO NOT correct y_true for magic number, just let it goes wrong and then times a correction terms
    def binary_accuracy_internal(y_true, y_pred):
        if from_logits:
            y_pred = keras.ops.sigmoid(y_pred)
        return keras.ops.mean(
            keras.ops.cast(
                keras.ops.equal(y_true, keras.ops.round(y_pred)),
                "float32",
            ),
            axis=-1,
        ) * magic_correction_term(y_true)

    if not from_logits:
        binary_accuracy_internal.__name__ = "binary_accuracy"  # set the name to be displayed in keras.ops/Keras log
    else:
        binary_accuracy_internal.__name__ = "binary_accuracy_from_logits"  # set the name to be displayed in keras.ops/Keras log

    return binary_accuracy_internal


def binary_accuracy(*args, **kwargs):
    """
    Calculate binary accuracy, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Binary accuracy
    :rtype: keras.ops.Tensor

    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    return __binary_accuracy(from_logits=False)(*args, **kwargs)


def binary_accuracy_from_logits(*args, **kwargs):
    """
    Calculate binary accuracy from logits, ignoring the magic number

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :return: Binary accuracy
    :rtype: keras.ops.Tensor

    :History: 2018-Jan-31 - Written - Henry Leung (University of Toronto)
    """
    return __binary_accuracy(from_logits=True)(*args, **kwargs)


def zeros_loss(y_true, y_pred, sample_weight=None):
    """
    Always return zeros

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param sample_weight: Sample weights
    :type sample_weight: Union(keras.ops.Tensor, keras.ops.Variable, list)
    :return: Zeros
    :rtype: keras.ops.Tensor
    :History: 2018-May-24 - Written - Henry Leung (University of Toronto)
    """
    losses = keras.ops.mean(
        keras.ops.where(
            magic_num_check(y_true), keras.ops.zeros_like(y_true), y_true
        )
        * 0.0
        + 0.0 * y_pred,
        axis=-1,
    )
    return weighted_loss(losses, sample_weight)


def median_error(y_true, y_pred, sample_weight=None, axis=-1):
    """
    Calculate median difference

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: keras.ops.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    # keras.ops.boolean_mask(keras.ops.logical_not(magic_num_check(y_true))
    return weighted_loss(median(y_true - y_pred, axis=axis), sample_weight)


def median_absolute_deviation(y_true, y_pred, sample_weight=None, axis=-1):
    """
    Calculate median absilute difference

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: keras.ops.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    return weighted_loss(
        median(keras.ops.abs(y_true - y_pred), axis=axis), sample_weight
    )


def mad_std(y_true, y_pred, sample_weight=None, axis=-1):
    """
    Calculate 1.4826 * median absilute difference

    :param y_true: Ground Truth
    :type y_true: Union(keras.ops.Tensor, keras.ops.Variable)
    :param y_pred: Prediction
    :type y_pred: Union(keras.ops.Tensor, keras.ops.Variable)
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: keras.ops.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    return weighted_loss(
        1.4826 * median_absolute_deviation(y_true, y_pred, axis=axis), sample_weight
    )


# Just alias functions
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
msle = mean_squared_logarithmic_error
me = mean_error
mpe = mean_percentage_error
mad = median_absolute_deviation

# legacy support
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
