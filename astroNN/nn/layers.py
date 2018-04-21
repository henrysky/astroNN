import math

import tensorflow as tf

from astroNN.config import keras_import_manager
from astroNN.nn import reduce_var

keras = keras_import_manager()
epsilon = keras.backend.epsilon
initializers = keras.initializers
Layer, Wrapper, InputSpec = keras.layers.Layer, keras.layers.Wrapper, keras.layers.InputSpec


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
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        mu, log_var = inputs
        kl_batch = - .5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MCDropout(Layer):
    """
    NAME: MCDropout
    PURPOSE: Dropout Layer for Bayesian Neural Network, this layer will always on regardless the learning phase flag
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, noise_shape=None, **kwargs):
        super().__init__(**kwargs)
        # tensorflow expects (0,1] retain prob
        self.rate = min(1. - epsilon(), max(0., rate))
        self.disable_layer = disable
        self.supports_masking = True
        self.noise_shape = noise_shape

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

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
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MCSpatialDropout1D(MCDropout):
    """
    NAME: MCSpatialDropout1D
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
        super().__init__(rate, disable, **kwargs)
        self.disable_layer = disable
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        return input_shape[0], 1, input_shape[2]


class MCSpatialDropout2D(MCDropout):
    """
    NAME: MCSpatialDropout2D
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
        super().__init__(rate, disable, **kwargs)
        self.disable_layer = disable
        self.input_spec = InputSpec(ndim=4)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        return input_shape[0], 1, 1, input_shape[3]


class MCGaussianDropout(Layer):
    """
    NAME: MCGaussianDropout
    PURPOSE: Dropout Layer for Bayesian Neural Network, this layer will always on regardless the learning phase flag
            standard deviation `sqrt(rate / (1 - rate))
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, **kwargs):
        super().__init__(**kwargs)
        self.rate = min(1. - epsilon(), max(0., rate))
        self.disable_layer = disable
        self.supports_masking = True
        self.rate = rate

    def call(self, inputs, training=None):
        stddev = math.sqrt(self.rate / (1.0 - self.rate))
        if self.disable_layer is True:
            return inputs
        else:
            return inputs * tf.random_normal(shape=tf.shape(inputs), mean=1.0, stddev=stddev)

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MCConcreteDropout(Wrapper):
    """
    :param layer: The layer to be applied concrete dropout
    :type layer: keras.layers.Layer
    :return: Layer Instance
    :rtype: instance
    """
    # """
    # NAME:
    #     McConcreteDropout
    # PURPOSE:
    #     Monte Carlo Dropout with Continuous Relaxation Layer Wrapper
    #     McConcreteDropout for Bayesian Neural Network, this layer will learn the dropout probability (arXiv:1705.07832)
    # INPUT:
    # OUTPUT:
    #     Output tensor
    # HISTORY:
    #     arXiv:1705.07832 By Yarin Gal, adapted from Yarin's original implementation
    #     2018-Mar-04 - Written - Henry Leung (University of Toronto)
    # """

    def __init__(self, layer, weight_regularizer=5e-13, dropout_regularizer=1e-4,
                 init_min=0.1, init_max=0.2, disable=False, **kwargs):
        """
        :param layer: The layer to be applied concrete dropout
        :type layer: keras.layers.Layer
        :return: Layer Instance
        :rtype: instance
        """
        assert 'kernel_regularizer' not in kwargs
        super().__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.disable_layer = disable
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = math.log(init_min) - math.log(1. - init_min)
        self.init_max = math.log(init_max) - math.log(1. - init_max)

    def build(self, input_shape=None):
        self.layer.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super().build()

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
        dropout_regularizer *= self.dropout_regularizer * tf.cast(input_dim, tf.float32)
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = {'rate': self.p.eval(session=keras.backend.get_session()),
                  'weight_regularizer': self.weight_regularizer, 'dropout_regularizer': self.dropout_regularizer}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

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
            return self.layer.call(inputs)
        else:
            return self.layer.call(self.concrete_dropout(inputs))


class MCBatchNorm(Layer):
    """
    NAME: MCBatchNorm
    PURPOSE: Batch Normalization Layer for Bayesian Neural Network
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Apr-12 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, disable=False, **kwargs):
        super().__init__(**kwargs)
        self.disable_layer = disable
        self.supports_masking = True
        self.epsilon = 1e-10

    def call(self, inputs, training=None):
        self.scale = tf.Variable(tf.ones([inputs.shape[-1]]))
        self.beta = tf.Variable(tf.zeros([inputs.shape[-1]]))
        self.mean = tf.Variable(tf.zeros([inputs.shape[-1]]), trainable=False)
        self.var = tf.Variable(tf.ones([inputs.shape[-1]]), trainable=False)

        if training is None:
            training = keras.backend.learning_phase()

        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        in_train = tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.beta, self.scale, self.epsilon)
        in_test = tf.nn.batch_normalization(inputs, self.mean, self.var, self.beta, self.scale, self.epsilon)

        return tf.where(tf.equal(training, True), in_train, in_test)

    def get_config(self):
        config = {'epsilon': self.epsilon}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

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
        super().__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training is None:
            training = keras.backend.learning_phase()

        noised = tf.add(inputs, tf.random_normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev))
        return tf.where(tf.equal(training, True), inputs, noised)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class FastMCInference():
    """
    NAME: FastMCInference
    PURPOSE:
        To create a model for fast MC Dropout Inference on GPU
    INPUT:
        number of monte carlo integration
        keras model
    OUTPUT:
        keras model
    HISTORY:
        2018-Apr-13 - Written - Henry Leung (University of Toronto)
    """
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, model):
        if type(model) == keras.Model or type(model) == keras.Sequential:
            self.model = model
        else:
            raise TypeError(f'FastMCInference expects keras Model, you gave {type(model)}')
        new_input = keras.layers.Input(shape=(self.model.input_shape[1:]), name='input')
        mc_model = keras.models.Model(inputs=self.model.inputs, outputs=self.model.outputs)
        mc = FastMCInferenceMeanVar()(keras.layers.TimeDistributed(mc_model)(FastMCRepeat(self.n)(new_input)))
        new_mc_model = keras.models.Model(inputs=new_input, outputs=mc)

        return new_mc_model


class FastMCInferenceMeanVar(Layer):
    """
    NAME: FastMCInferenceMeanVar
    PURPOSE:
        Take mean and variance of the results of a TimeDistributed layer, assuming axis=1 is the timestamp axis
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-02 - Written - Henry Leung (University of Toronto)
        2018-Apr-13 - Update - Henry Leung (University of Toronto)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return 2, input_shape[0], input_shape[2:]

    def get_config(self):
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def call(self, inputs, training=None):
        # need to stack because keras can only handle one output
        mean, var = tf.nn.moments(inputs, axes=1)
        return tf.squeeze(tf.stack([[mean], [var]], axis=-1))


class FastMCRepeat(Layer):
    """
    NAME: FastMCRepeat
    PURPOSE: Prepare data to do inference, Repeats the input n times at axis=1
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Mar-05 - Written - Henry Leung (University of Toronto)
        2018-Apr-13 - Update - Henry Leung (University of Toronto)
    """

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n) + (input_shape[1:])

    def call(self, inputs, training=None):
        expanded_inputs = tf.expand_dims(inputs, 1)
        # we want [1, self.n, 1.....]
        return tf.tile(expanded_inputs, tf.concat([[1, self.n], tf.ones_like(tf.shape(expanded_inputs))[2:]], axis=0))

    def get_config(self):
        config = {'n': self.n}
        base_config = super().get_config()
        return {**base_config.items(), **config}
