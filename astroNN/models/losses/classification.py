# -----------------------------------------------------------------------#
#   astroNN.models.losses.classification: losses function for classification
# ----------------------------------------------------------------------#
import keras.backend as K
from tensorflow.contrib import distributions
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from astroNN import MAGIC_NUMBER


def astronn_sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None):
    """
    NAME: astronn_sigmoid_cross_entropy_with_logits
    PURPOSE: Computes sigmoid cross entropy given `logits`.
             # Measures the probability error in discrete classification tasks in which each
             class is independent and not mutually exclusive.  For instance, one could
             perform multilabel classification where a picture can contain both an elephant
             and a dog at the same time.
    INPUT:
    OUTPUT:
          A `Tensor` of the same shape as `logits` with the componentwise
          logistic losses.
    HISTORY:
        2018-Jan-18 - Written - Henry Leung (University of Toronto)
    """
    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        labels = ops.convert_to_tensor(labels, name="labels")
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (logits.get_shape(), labels.get_shape()))

        zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)

        magic_cond = (labels == MAGIC_NUMBER)  # To deal with missing labels
        return array_ops.where(magic_cond, zeros,
                               math_ops.add(relu_logits - logits * labels, math_ops.log1p(math_ops.exp(neg_abs_logits)))
                               , name=name)


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
    if not from_logits:
        # Deal with magic number first
        y_true = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), y_true)
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
        # manual computation of crossentropy
        _epsilon = K.tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = K.tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - K.tf.reduce_sum(y_true * K.tf.log(y_pred), len(y_pred.get_shape()) - 1)
    else:
        try:
            return K.tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        except AttributeError or ImportError:
            return K.tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


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
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # Deal with magic number first
        y_true = K.tf.where(K.tf.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), y_true)
        # transform back to logits
        _epsilon = K.tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = K.tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = K.tf.log(y_pred / (1 - y_pred))

    return K.mean(astronn_sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)


def gaussian_crossentropy(true, pred, dist, undistorted_loss, num_classes):
    # for a single monte carlo simulation,
    #   calculate categorical_crossentropy of
    #   predicted logit values plus gaussian
    #   noise vs true values.
    # true - true values. Shape: (N, C)
    # pred - predicted logit values. Shape: (N, C)
    # dist - normal distribution to sample from. Shape: (N, C)
    # undistorted_loss - the crossentropy losses without variance distortion. Shape: (N,)
    # num_classes - the number of classes. C
    # returns - total differences for all classes (N,)
    def map_fn(i):
        std_samples = K.transpose(dist.sample(num_classes))
        distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -K.elu(diff)

    return map_fn


def bayes_crossentropy_wrapper():
    # Bayesian categorical cross entropy.
    # N data points, C classes, T monte carlo simulations
    # true - true values. Shape: (N, C)
    # pred_var - predicted logit values and variance. Shape: (N, C + 1)
    # returns - losses (N,)
    def bayes_crossentropy(true, pred_var):
        T = 20
        num_classes = K.shape(pred_var)[1]
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes:])
        # shape: (N,)
        variance = pred_var[:, num_classes]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
        # shape: (T,)
        iterable = K.variable(K.tf.ones(T))
        dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
        monte_carlo_results = K.map_fn(
            gaussian_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable,
            name='monte_carlo_results')

        variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

        return variance_loss + undistorted_loss + variance_depressor

    return bayes_crossentropy
