import keras.backend as K
from keras.layers import Layer
import tensorflow as tf


class KLDivergenceLayer(Layer):
    """
    NAME: KLDivergenceLayer
    PURPOSE:
        Identity transform layer that adds KL divergence to the final model losses.
        KL divergence used to force the latent space match the prior (in this case its unit gaussian)
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        mu, log_var = inputs
        kl_batch = - .5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        config = {'None': None}
        base_config = super(KLDivergenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BayesianDropout(Layer):
    """
    NAME: BayesianDropout
    PURPOSE: Dropout Layer for Bayeisna Neural Network, this layer will always regardless the learning phase flag
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, **kwargs):
        super(BayesianDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.supports_masking = True

    def call(self, inputs, training=None):
        retain_prob = 1. - self.rate
        return tf.nn.dropout(inputs * 1., retain_prob)

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(BayesianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ErrorProp(Layer):
    """
    NAME: ErrorProp
    PURPOSE: Propagate Error Layer, do nothing during training, add gaussian noise during testing phase
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, stddev, **kwargs):
        super(ErrorProp, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + tf.random_normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev)

        return K.in_train_phase(inputs, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(ErrorProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class TimeDistributedMeanVar(Layer):
    """
    NAME: TimeDistributedMeanVar
    PURPOSE: Take mean and variance of the results of a TimeDistributed layer.
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-02 - Written - Henry Leung (University of Toronto)
    """

    def build(self, input_shape):
        super(TimeDistributedMeanVar, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]

    def call(self, x, training=None):
        return tf.reduce_mean(x, axis=1), K.var(x, axis=1)
