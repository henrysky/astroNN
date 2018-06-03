import math

import tensorflow as tf
from astroNN.config import keras_import_manager

keras = keras_import_manager()
epsilon = keras.backend.epsilon
initializers = keras.initializers
Layer, Wrapper, InputSpec = keras.layers.Layer, keras.layers.Wrapper, keras.layers.InputSpec


class KLDivergenceLayer(Layer):
    """
    | Identity transform layer that adds KL divergence to the final model losses.
    | KL divergence used to force the latent space match the prior (in this case its unit gaussian)


    :return: A layer
    :rtype: object
    :History: 2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, name=None, **kwargs):
        self.is_placeholder = True
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied, concatenated tf.tensor of mean and std in latent space
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        mu, log_var = inputs
        kl_batch = - .5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MCDropout(Layer):
    """
    Dropout Layer for Bayesian Neural Network, this layer will always on regardless the learning phase flag

    :param rate: Dropout Rate between 0 and 1
    :type rate: float
    :param disable: Dropout on or off
    :type disable: boolean
    :return: A layer
    :rtype: object
    :History: 2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, noise_shape=None, name=None, **kwargs):
        # tensorflow expects (0,1] retain prob
        self.rate = min(1. - epsilon(), max(0., rate))
        self.disable_layer = disable
        self.supports_masking = True
        self.noise_shape = noise_shape
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        retain_prob = 1. - self.rate
        noise_shape = self._get_noise_shape(inputs)
        if self.disable_layer is True:
            return inputs
        else:
            return tf.nn.dropout(inputs * 1., retain_prob, noise_shape)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MCSpatialDropout1D(MCDropout):
    """
    Spatial 1D version of Dropout of Dropout Layer for Bayesian Neural Network,
    this layer will always regardless the learning phase flag

    :param rate: Dropout Rate between 0 and 1
    :type rate: float
    :param disable: Dropout on or off
    :type disable: boolean
    :return: A layer
    :rtype: object
    :History: 2018-Mar-07 - Written - Henry Leung (University of Toronto)
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
    Spatial 2D version of Dropout of Dropout Layer for Bayesian Neural Network,
    this layer will always regardless the learning phase flag

    :param rate: Dropout Rate between 0 and 1
    :type rate: float
    :param disable: Dropout on or off
    :type disable: boolean
    :return: A layer
    :rtype: object
    :History: 2018-Mar-07 - Written - Henry Leung (University of Toronto)
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
    Dropout Layer for Bayesian Neural Network, this layer will always on regardless the learning phase flag
    standard deviation sqrt(rate / (1 - rate))

    :param rate: Dropout Rate between 0 and 1
    :type rate: float
    :param disable: Dropout on or off
    :type disable: boolean
    :return: A layer
    :rtype: object
    :History: 2018-Mar-07 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, rate, disable=False, name=None, **kwargs):
        self.rate = min(1. - epsilon(), max(0., rate))
        self.disable_layer = disable
        self.supports_masking = True
        self.rate = rate
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        stddev = math.sqrt(self.rate / (1.0 - self.rate))
        if self.disable_layer is True:
            return inputs
        else:
            return inputs * tf.random_normal(shape=tf.shape(inputs), mean=1.0, stddev=stddev)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'rate': self.rate}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class MCConcreteDropout(Wrapper):
    """
    | Monte Carlo Dropout with Continuous Relaxation Layer Wrapper This layer will learn the dropout probability
    | arXiv:1705.07832

    :param layer: The layer to be applied concrete dropout
    :type layer: keras.layers.Layer
    :return: A layer
    :rtype: object
    :History: 2018-Mar-04 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, layer, weight_regularizer=5e-13, dropout_regularizer=1e-4,
                 init_min=0.1, init_max=0.2, disable=False, **kwargs):
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
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
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
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        if self.disable_layer is True:
            return self.layer.call(inputs)
        else:
            return self.layer.call(self.concrete_dropout(inputs))


class MCBatchNorm(Layer):
    """
    Monte Carlo Batch Normalization Layer for Bayesian Neural Network

    :param disable: Dropout on or off
    :type disable: boolean
    :return: A layer
    :rtype: object
    :History: 2018-Apr-12 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, disable=False, name=None, **kwargs):
        self.disable_layer = disable
        self.supports_masking = True
        self.epsilon = 1e-10
        self.scale = None
        self.beta = None
        self.mean = None
        self.var = None
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
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
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'epsilon': self.epsilon}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class ErrorProp(Layer):
    """
    Propagate Error Layer, do nothing during training, add gaussian noise during testing phase

    :param stddev: Known 1-S.D. Uncertainty in input data
    :type stddev: float
    :return: A layer
    :rtype: object
    :History: 2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, stddev, name=None, **kwargs):
        self.supports_masking = True
        self.stddev = stddev
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        if training is None:
            training = keras.backend.learning_phase()

        noised = tf.add(inputs, tf.random_normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev))
        return tf.where(tf.equal(training, True), inputs, noised)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'stddev': self.stddev}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class FastMCInference():
    """
    To create a model for fast MC Dropout Inference on GPU

    :param n: Number of Monte Carlo integration
    :type n: int
    :return: A layer
    :rtype: object
    :History: 2018-Apr-13 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, n, **kwargs):
        self.n = n

    def __call__(self, model):
        """
        :param model: Keras model to be accelerated
        :type model: Union[keras.Model, keras.Sequential]
        :return: Accelerated Keras model
        :rtype: Union[keras.Model, keras.Sequential]
        """
        if isinstance(model, keras.Model) or isinstance(model, keras.Sequential):
            self.model = model
        else:
            raise TypeError(f'FastMCInference expects keras Model, you gave {type(model)}')
        new_input = keras.layers.Input(shape=(self.model.input_shape[1:]), name='input')
        mc_model = keras.models.Model(inputs=self.model.inputs, outputs=self.model.outputs)
        mc = FastMCInferenceMeanVar()(keras.layers.TimeDistributed(mc_model)(FastMCRepeat(self.n)(new_input)))
        new_mc_model = keras.models.Model(inputs=new_input, outputs=mc)

        return new_mc_model

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'n': self.n}
        return config


class FastMCInferenceMeanVar(Layer):
    """
    Take mean and variance of the results of a TimeDistributed layer, assuming axis=1 is the timestamp axis

    :return: A layer
    :rtype: object
    :History:
        | 2018-Feb-02 - Written - Henry Leung (University of Toronto)
        | 2018-Apr-13 - Update - Henry Leung (University of Toronto)
    """

    def __init__(self, name=None, **kwargs):
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        return 2, input_shape[0], input_shape[2:]

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        # need to stack because keras can only handle one output
        mean, var = tf.nn.moments(inputs, axes=1)
        return tf.stack((tf.squeeze([mean]), tf.squeeze([var])), axis=-1)


class FastMCRepeat(Layer):
    """
    Prepare data to do inference, Repeats the input n times at axis=1

    :param n: Number of Monte Carlo integration
    :type n: int
    :return: A layer
    :rtype: object
    :History:
        | 2018-Feb-02 - Written - Henry Leung (University of Toronto)
        | 2018-Apr-13 - Update - Henry Leung (University of Toronto)
    """

    def __init__(self, n, name=None, **kwargs):
        self.n = n
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n) + (input_shape[1:])

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer which is the repeated Tensor
        :rtype: tf.Tensor
        """
        expanded_inputs = tf.expand_dims(inputs, 1)
        # we want [1, self.n, 1.....]
        return tf.tile(expanded_inputs, tf.concat([[1, self.n], tf.ones_like(tf.shape(expanded_inputs))[2:]], axis=0))

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'n': self.n}
        base_config = super().get_config()
        return {**base_config.items(), **config}


class StopGrad(Layer):
    """
    Stop gradient backpropagation via this layer during training, act as an identity layer during testing

    :return: A layer
    :rtype: object
    :History: 2018-May-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, name=None, **kwargs):
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer which is just the original tensor
        :rtype: tf.Tensor
        """
        if training:
            return tf.stop_gradient(inputs)
        else:
            return inputs

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}


class BoolMask(Layer):
    """
    Boolean Masking layer

    :return: A layer
    :rtype: object
    :History: 2018-May-28 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, mask, name=None, **kwargs):
        self.boolmask = mask
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        return tuple((input_shape[0], self.boolmask.sum()))

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer which is just the masked tensor
        :rtype: tf.Tensor
        """
        return tf.boolean_mask(inputs, self.boolmask, axis=1)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {'None': None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}
