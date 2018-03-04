import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import initializers
from keras.backend import epsilon
from keras.engine import InputSpec
from keras.layers import Layer, Wrapper


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
    PURPOSE: Dropout Layer for Bayesian Neural Network, this layer will always regardless the learning phase flag
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, **kwargs):
        super(BayesianDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.disable_layer = disable
        self.supports_masking = True

    def call(self, inputs, training=None):
        retain_prob = 1. - self.rate
        if self.disable_layer is True:
            return inputs
        else:
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


class ConcreteDropout(Wrapper):
    """
    NAME: ConcreteDropout
    PURPOSE: ConcreteDropout for Bayesian Neural Network, this layer will learn the dropout probability (arXiv:1705.07832)
    INPUT:
    OUTPUT:
        Output tensor
    HISTORY:
        arXiv:1705.07832 By Yarin Gal, adapted from Yarin's original implementation
        2018-Mar-04 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.2, disable=False, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.disable_layer = disable
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit', shape=(1,),
                                             initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                             trainable=True)
        self.p = tf.nn.sigmoid(self.p_logit[0])

        # initialise regularizer / prior KL term
        input_dim = np.prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * tf.log(self.p)
        dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        eps = epsilon()

        unif_noise = tf.random_uniform(shape=K.shape(x))
        drop_prob = (tf.log(self.p + eps) - tf.log(1. - self.p + eps) + tf.log(unif_noise + eps) - tf.log(
            1. - unif_noise + eps))
        drop_prob = tf.nn.sigmoid(drop_prob / 0.1)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.disable_layer is True:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            self.layer.call(inputs)
