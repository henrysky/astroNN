# -----------------------------------------------------------------------#
#   astroNN.models.loss.classification: loss function for classification
# ----------------------------------------------------------------------#
import keras.backend as K
from tensorflow.contrib import distributions

from astroNN import MAGIC_NUMBER


def categorical_cross_entropy(y_true, y_pred):
    return K.sum(K.switch(K.equal(y_true, MAGIC_NUMBER), K.tf.zeros_like(y_true), y_true * K.log(y_pred)), axis=-1)


def categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.

    # Returns
        Output tensor.
    """
    # TODO: Add magic number support
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= K.tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        _epsilon = K.tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = K.tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - K.tf.reduce_sum(target * K.tf.log(output), axis=len(output.get_shape()) - 1)
    else:
        return K.tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


def gaussian_crossentropy(true, pred, dist, undistorted_loss, num_classes):
    # for a single monte carlo simulation,
    #   calculate categorical_crossentropy of
    #   predicted logit values plus gaussian
    #   noise vs true values.
    # true - true values. Shape: (N, C)
    # pred - predicted logit values. Shape: (N, C)
    # dist - normal distribution to sample from. Shape: (N, C)
    # undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
    # num_classes - the number of classes. C
    # returns - total differences for all classes (N,)
    def map_fn(i):
        std_samples = K.transpose(dist.sample(num_classes))
        distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -K.elu(diff)

    return map_fn


def bayes_crossentropy_wrapper(T, num_classes):
    # Bayesian categorical cross entropy.
    # N data points, C classes, T monte carlo simulations
    # true - true values. Shape: (N, C)
    # pred_var - predicted logit values and variance. Shape: (N, C + 1)
    # returns - loss (N,)
    def bayes_crossentropy(true, pred_var):
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
