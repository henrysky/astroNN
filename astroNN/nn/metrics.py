# ---------------------------------------------------------------#
#   astroNN.nn.metrics: metrics
# ---------------------------------------------------------------#
import tensorflow as tf

from astroNN.nn.losses import binary_accuracy, binary_accuracy_from_logits
from astroNN.nn.losses import categorical_accuracy
from astroNN.nn.losses import mean_absolute_error
from astroNN.nn.losses import mean_absolute_percentage_error
from astroNN.nn.losses import mean_error
from astroNN.nn.losses import mean_percentage_error
from astroNN.nn.losses import mean_squared_error
from astroNN.nn.losses import mean_squared_logarithmic_error
from astroNN.nn.losses import weighted_loss

# Just alias functions
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
msle = mean_squared_logarithmic_error
me = mean_error
mpe = mean_percentage_error
categorical_accuracy = categorical_accuracy
binary_accuracy = binary_accuracy
binary_accuracy_from_logits = binary_accuracy_from_logits


def median(x, axis=None):
    """
    Calculate median
    
    :param x: Data
    :type x: tf.Tensor
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: tf.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    @tf.function
    def median_internal(_x):
        half_shape = tf.shape(_x)[0] // 2
        _median = tf.nn.top_k(_x, half_shape).values[-1]
        return _median
        
    if axis is None:
        x_flattened = tf.reshape(x, [-1])
        median = median_internal(x_flattened)
        return median
    else:
        x_unstacked = tf.unstack(tf.transpose(x), axis=axis)
        median = tf.stack([median_internal(_x) for _x in x_unstacked])
        return median

def median_error(x, y, sample_weight=None):
    """
    Calculate median difference
    
    :param x: Data
    :type x: tf.Tensor
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: tf.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    return weighted_loss(median(x - y, axis=None), sample_weight)

def median_absolute_deviation(x, y, sample_weight=None):
    """
    Calculate median absilute difference
    
    :param x: Data
    :type x: tf.Tensor
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: tf.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    return weighted_loss(median(tf.abs(x - y), axis=None), sample_weight)

def mad_std(x, y, sample_weight=None):
    """
    Calculate 1.4826 * median absilute difference
    
    :param x: Data
    :type x: tf.Tensor
    :param axis: Axis
    :type axis: int
    :return: Variance
    :rtype: tf.Tensor
    :History: 2021-Aug-13 - Written - Henry Leung (University of Toronto)
    """
    return weighted_loss(1.4826 * median_absolute_deviation(x, y), sample_weight)


mad = median_absolute_deviation
