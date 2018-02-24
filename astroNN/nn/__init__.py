def magic_correction_term(y_true):
    """
    NAME: magic_correction_term
    PURPOSE: calculate a correction term to prevent the loss being lowered by magic_num, since we assume
    whatever neural network is predicting, its right for those magic number
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-30 - Written - Henry Leung (University of Toronto)
        2018-Feb-17 - Written - Henry Leung (University of Toronto)
    """
    import tensorflow as tf
    from astroNN import MAGIC_NUMBER

    num_nonzero = tf.reduce_sum(tf.cast(tf.not_equal(y_true, MAGIC_NUMBER), tf.float32), axis=-1)
    num_zero = tf.reduce_sum(tf.cast(tf.equal(y_true, MAGIC_NUMBER), tf.float32), axis=-1)

    # If no magic number, then num_zero=0 and whole expression is just 1 and get back our good old loss
    # If num_nonzero is 0, that means we don't have any information, then set the correction term to ones
    return (num_nonzero + num_zero) / num_nonzero
