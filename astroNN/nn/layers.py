import math

import tensorflow as tf

from astroNN import keras_import_manager

keras = keras_import_manager()
epsilon, in_train_phase = keras.backend.epsilon, keras.backend.in_train_phase
initializers = keras.initializers
Layer, Wrapper, InputSpec = keras.layers.Layer, keras.layers.Wrapper, keras.layers.InputSpec
keras.layers.Dropout

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
        noise_shape = self._get_noise_shape(inputs)
        if self.disable_layer is True:
            return inputs
        else:
            return tf.nn.dropout(inputs * 1., retain_prob, noise_shape)

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape}
        base_config = super(BayesianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BayesianSpatialDropout1D(BayesianDropout):
    """
    NAME: BayesianSpatialDropout1D
    PURPOSE:
        Spatial 1D version of Dropout of Dropout Layer for Bayesian Neural Network,
        this layer will always regardless the learning phase flag
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-07 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, **kwargs):
        super(BayesianSpatialDropout1D, self).__init__(rate, disable, **kwargs)
        self.disable_layer = disable
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape


class BayesianSpatialDropout2D(BayesianDropout):
    """
    NAME: BayesianSpatialDropout2D
    PURPOSE:
        Spatial 1D version of Dropout of Dropout Layer for Bayesian Neural Network,
        this layer will always regardless the learning phase flag
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-07 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, **kwargs):
        super(BayesianSpatialDropout2D, self).__init__(rate, disable, **kwargs)
        self.disable_layer = disable
        self.input_spec = InputSpec(ndim=4)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        return input_shape[0], 1, 1, input_shape[3]


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

        return in_train_phase(inputs, noised, training=training)

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
        # 2 is mean and var, input_shape thingys are the input shape
        return 2, input_shape[0], input_shape[2:],

    def get_config(self):
        config = {'None': None}
        base_config = super(TimeDistributedMeanVar, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, training=None):
        # need to stack because keras can only handle one output
        return tf.stack(tf.nn.moments(x, axes=1))


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
        self.init_min = math.log(init_min) - math.log(1. - init_min)
        self.init_max = math.log(init_max) - math.log(1. - init_max)

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
        input_dim = tf.reduce_prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * tf.log(self.p)
        dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = {'rate': self.p, 'weight_regularizer': self.weight_regularizer,
                  'dropout_regularizer': self.dropout_regularizer}
        base_config = super(ConcreteDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def concrete_dropout(self, x):
        eps = epsilon()

        unif_noise = tf.random_uniform(shape=tf.shape(x))
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


class BayesianRepeatVector(Layer):
    """
    NAME: BayesianRepeatVector
    PURPOSE: Repeats the input n times during testing time, do nothing during training time
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-05 - Written - Henry Leung (University of Toronto)
    """
    def __init__(self, n, **kwargs):
        super(BayesianRepeatVector, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, inputs, training=None):
        return in_train_phase(inputs, tf.tile(tf.expand_dims(inputs, 1), tf.stack([1, self.n, 1])), training=training)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(BayesianRepeatVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
