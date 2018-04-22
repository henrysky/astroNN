def magic_correction_term(y_true):
    """
    Calculate a correction term to prevent the loss being lowered by magic_num

    :param y_true: Ground Truth
    :type y_true: tf.Tensor
    :return: Correction Term
    :rtype: tf.Tensor
    :History:
        | 2018-Jan-30 - Written - Henry Leung (University of Toronto)
        | 2018-Feb-17 - Updated - Henry Leung (University of Toronto)
    """
    import tensorflow as tf
    from astroNN.config import MAGIC_NUMBER

    num_nonmagic = tf.reduce_sum(tf.cast(tf.not_equal(y_true, MAGIC_NUMBER), tf.float32), axis=-1)
    num_magic = tf.reduce_sum(tf.cast(tf.equal(y_true, MAGIC_NUMBER), tf.float32), axis=-1)

    # If no magic number, then num_zero=0 and whole expression is just 1 and get back our good old loss
    # If num_nonzero is 0, that means we don't have any information, then set the correction term to ones
    return (num_nonmagic + num_magic) / num_nonmagic


def reduce_var(x, axis=None, keepdims=False):
    """
    Calculate variance using Tensorflow (as opposed to tf.nn.moment which return both variance and mean)

    :param x: Data
    :type x: tf.Tensor
    :param axis: Axis
    :type axis: int
    :param keepdims: Keeping variance dimension as data or not
    :type keepdims: boolean
    :return: Variance
    :rtype: tf.Tensor
    :History: 2018-Mar-04 - Written - Henry Leung (University of Toronto)
    """
    import tensorflow as tf

    m = tf.reduce_mean(x, axis, True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis, keepdims)