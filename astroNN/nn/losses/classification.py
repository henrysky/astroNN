# -----------------------------------------------------------------------#
#   astroNN.models.losses.classification: losses function for classification
# ----------------------------------------------------------------------#
import tensorflow as tf
from keras.backend import epsilon

from astroNN import MAGIC_NUMBER
from astroNN.nn import magic_correction_term


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
        try:
            return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred) * \
                   magic_correction_term(y_true)
        except AttributeError or ImportError:
            return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * \
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


def bayesian_crossentropy_wrapper(from_logits=True):
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
        2018-Feb-09 - Written - Henry Leung (University of Toronto)
    """

    # TODO: need working and review
    def bayesian_crossentropy(y_true, y_pred):
        T = 25
        num_classes = tf.shape(y_pred)[1]
        std = tf.sqrt(y_pred)
        variance = y_pred[:, num_classes]
        variance_depressor = tf.exp(variance) - tf.ones_like(variance)
        pred = y_pred[:, 0:num_classes]
        undistorted_loss = categorical_cross_entropy(pred, y_true, from_logits=from_logits)
        iterable = tf.ones(T)
        norm_dist = tf.random_normal(shape=tf.shape(std), mean=tf.zeros_like(std), stddev=std)
        monte_carlo_results = tf.map_fn(
            gaussian_crossentropy(y_true, pred, norm_dist, undistorted_loss, num_classes), iterable,
            name='monte_carlo_results')

        variance_loss = tf.reduce_mean(monte_carlo_results, axis=0) * undistorted_loss

        return (variance_loss + undistorted_loss + variance_depressor) * magic_correction_term(y_true)

    return bayesian_crossentropy


def gaussian_crossentropy(true, pred, dist, undistorted_loss, num_classes):
    """
    NAME: gaussian_crossentropy
    PURPOSE: gaussian
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-09 - Written - Henry Leung (University of Toronto)
    """

    # TODO: need working and review
    def map_fn(i):
        std_samples = tf.transpose(dist.sample(num_classes))
        distorted_loss = categorical_cross_entropy(pred + std_samples, true, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -tf.nn.elu(diff)

    return map_fn
