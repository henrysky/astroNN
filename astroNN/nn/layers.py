import math
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.parallel_for.control_flow_ops import pfor

from astroNN.nn import intpow_avx2

# from tensorflow_probability.python.layers import util as tfp_layers_util
# from tensorflow_probability.python.layers.dense_variational import _DenseVariational as DenseVariational_Layer
# from tensorflow_probability.python.math import random_rademacher

epsilon = tfk.backend.epsilon
initializers = tfk.initializers
activations = tfk.activations
Layer, Wrapper, InputSpec = tfk.layers.Layer, tfk.layers.Wrapper, tfk.layers.InputSpec


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
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
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
        kl_batch = -0.5 * tf.reduce_sum(
            1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"None": None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class VAESampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def __init__(self, name=None, **kwargs):
        self.supports_masking = True
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


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
        self.rate = rate
        self.disable_layer = disable
        self.supports_masking = True
        self.noise_shape = noise_shape
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        noise_shape = self._get_noise_shape(inputs)
        if self.disable_layer is True:
            return inputs
        else:
            return tf.nn.dropout(x=inputs, rate=self.rate, noise_shape=noise_shape)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"rate": self.rate, "noise_shape": self.noise_shape}
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
        self.rate = min(1.0 - epsilon(), max(0.0, rate))
        self.disable_layer = disable
        self.supports_masking = True
        self.rate = rate
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
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
            return inputs * tf.random.normal(
                shape=tf.shape(inputs), mean=1.0, stddev=stddev
            )

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"rate": self.rate}
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

    def __init__(
        self,
        layer,
        weight_regularizer=5e-13,
        dropout_regularizer=1e-4,
        init_min=0.1,
        init_max=0.2,
        disable=False,
        **kwargs,
    ):
        assert "kernel_regularizer" not in kwargs
        super().__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.disable_layer = disable
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = math.log(init_min) - math.log(1.0 - init_min)
        self.init_max = math.log(init_max) - math.log(1.0 - init_max)

    def build(self, input_shape=None):
        self.layer.input_spec = InputSpec(shape=input_shape)
        if hasattr(self.layer, "built") and not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        # initialise p
        self.p_logit = self.add_weight(
            name="p_logit",
            shape=(1,),
            initializer=initializers.RandomUniform(self.init_min, self.init_max),
            dtype=tf.float32,
            trainable=True,
        )
        self.p = tf.nn.sigmoid(self.p_logit)
        tf.compat.v1.add_to_collection("LAYER_P", self.p)

        # initialise regularizer / prior KL term
        input_dim = tf.reduce_prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = (
            self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1.0 - self.p)
        )
        dropout_regularizer = self.p * tf.math.log(self.p)
        dropout_regularizer += (1.0 - self.p) * tf.math.log(1.0 - self.p)
        dropout_regularizer *= self.dropout_regularizer * tf.cast(input_dim, tf.float32)
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)
        # Add the regularisation loss to collection.
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, regularizer
        )

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {
            "rate": tf.nn.sigmoid(self.p_logit).numpy(),
            "weight_regularizer": self.weight_regularizer,
            "dropout_regularizer": self.dropout_regularizer,
        }
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def concrete_dropout(self, x):
        eps = epsilon()
        self.p_call = tf.nn.sigmoid(self.p_logit)

        unif_noise = tf.random.uniform(shape=tf.shape(x))
        drop_prob = (
            tf.math.log(self.p_call + eps)
            - tf.math.log(1.0 - self.p_call + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1.0 - unif_noise + eps)
        )
        drop_prob = tf.nn.sigmoid(drop_prob / 0.1)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - self.p_call
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
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
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
            training = tfk.backend.learning_phase()

        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        in_train = tf.nn.batch_normalization(
            inputs, batch_mean, batch_var, self.beta, self.scale, self.epsilon
        )
        in_test = tf.nn.batch_normalization(
            inputs, self.mean, self.var, self.beta, self.scale, self.epsilon
        )

        output_tensor = tf.where(tf.equal(training, True), in_train, in_test)
        output_tensor._uses_learning_phase = True
        return output_tensor

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"epsilon": self.epsilon}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class ErrorProp(Layer):
    """
    Propagate Error Layer by adding gaussian noise (mean=0, std=err) during testing phase from ``input_err`` tensor

    :return: A layer
    :rtype: object
    :History: 2018-Feb-05 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, name=None, **kwargs):
        self.supports_masking = True
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: a list of Tensor which [input_tensor, input_error_tensor]
        :type inputs: list[tf.Tensor]

        :return: Tensor after applying the layer
        :rtype: tf.Tensor
        """
        if training is None:
            training = tfk.backend.learning_phase()

        noised = tf.random.normal([1], mean=inputs[0], stddev=inputs[1])
        output_tensor = tf.where(tf.equal(training, True), inputs[0], noised)
        output_tensor._uses_learning_phase = True
        return output_tensor

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class FastMCInference:
    """
    Turn a model for fast Monte Carlo (Dropout, Flipout, etc) Inference on GPU

    :param n: Number of Monte Carlo integration
    :type n: int
    :return: A layer
    :rtype: object
    :History:
        | 2018-Apr-13 - Written - Henry Leung (University of Toronto)
        | 2021-Apr-14 - Updated - Henry Leung (University of Toronto)

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
        if isinstance(model, tfk.Model) or isinstance(model, tfk.Sequential):
            self.model = model
        else:
            raise TypeError(
                f"FastMCInference expects tensorflow.keras Model, you gave {type(model)}"
            )
        new_input = tfk.layers.Input(shape=(self.model.input_shape[1:]), name="input")
        mc_model = tfk.models.Model(
            inputs=self.model.inputs, outputs=self.model.outputs
        )

        mc = FastMCInferenceMeanVar()(
            FastMCInferenceV2_internal(mc_model, self.n)(new_input)
        )
        new_mc_model = tfk.models.Model(inputs=new_input, outputs=mc)

        return new_mc_model

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"n": self.n}
        return config


class FastMCInferenceV2_internal(Wrapper):
    def __init__(self, model, n=100, **kwargs):
        if isinstance(model, tfk.Model) or isinstance(model, tfk.Sequential):
            self.layer = model
            self.n = n
        else:
            raise TypeError(
                f"FastMCInference expects tensorflow.keras Model, you gave {type(model)}"
            )

        super(FastMCInferenceV2_internal, self).__init__(model, **kwargs)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.layer.output_shape

    def call(self, inputs, training=None, mask=None):
        def loop_fn(i):
            return self.layer(inputs)

        outputs = pfor(loop_fn, self.n, parallel_iterations=self.n)
        return outputs


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
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        return 2, input_shape[0], input_shape[2:]

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"None": None}
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
        mean, var = tf.nn.moments(inputs, axes=0)
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
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
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
        return tf.tile(
            expanded_inputs,
            tf.concat(
                [[1, self.n], tf.ones_like(tf.shape(expanded_inputs))[2:]], axis=0
            ),
        )

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"n": self.n}
        base_config = super().get_config()
        return {**base_config.items(), **config}


class StopGrad(Layer):
    """
    Stop gradient backpropagation via this layer during training, act as an identity layer during testing by default.

    :param always_on: Default False which will on during train time and off during test time. True to enable it in every situation
    :type always_on: bool
    :return: A layer
    :rtype: object
    :History: 2018-May-23 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, name=None, always_on=False, **kwargs):
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)
        self.always_on = always_on

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
        if self.always_on:
            return tf.stop_gradient(inputs)
        else:
            if training is None:
                training = tfk.backend.learning_phase()
            output_tensor = tf.where(
                tf.equal(training, True), tf.stop_gradient(inputs), inputs
            )
            output_tensor._uses_learning_phase = True
            return output_tensor

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"None": None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}


class BoolMask(Layer):
    """
    Boolean Masking layer, please notice it is best to flatten input before using BoolMask

    :param mask: numpy boolean array as a mask for incoming tensor
    :type mask: np.ndarray
    :return: A layer
    :rtype: object
    :History: 2018-May-28 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, mask, name=None, **kwargs):
        if sum(mask) == 0:
            raise ValueError("The mask is all False, which is invalid")
        else:
            self.boolmask = mask
        self.mask_shape = self.boolmask.sum()
        self.supports_masking = True
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.mask_shape)

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer which is just the masked tensor
        :rtype: tf.Tensor
        """
        batchsize = tf.shape(inputs)[0]
        # need to reshape because tf.keras cannot get the Tensor shape correctly from tf.boolean_mask op
        return tf.reshape(
            tf.boolean_mask(inputs, self.boolmask, axis=1), [batchsize, self.mask_shape]
        )

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"None": None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}


class PolyFit(Layer):
    """
    n-deg polynomial fitting layer which acts as an neural network layer to be optimized

    :param deg: degree of polynomial
    :type deg: int
    :param output_units: number of output neurons
    :type output_units: int
    :param use_xbias: If True, then fitting output=P(inputs)+inputs, else fitting output=P(inputs)
    :type use_xbias: bool
    :param init_w: [Optional] list of initial weights if there is any, the list should be [n-degree, input_size, output_size]
    :type init_w: Union[NoneType, list]
    :param name: [Optional] name of the layer
    :type name: Union[NoneType, str]
    :param activation: [Optional] activation, default is 'linear'
    :type activation: Union[NoneType, str]
    :param kernel_regularizer: [Optional] kernel regularizer
    :type kernel_regularizer: Union[NoneType, str]
    :param kernel_constraint: [Optional] kernel constraint
    :type kernel_constraint: Union[NoneType, str]
    :return: A layer
    :rtype: object
    :History: 2018-Jul-24 - Written - Henry Leung (University of Toronto)
    """

    def __init__(
        self,
        deg=1,
        output_units=1,
        use_xbias=True,
        init_w=None,
        name=None,
        activation=None,
        kernel_regularizer=None,
        kernel_constraint=None,
    ):
        super().__init__(name=name)
        self.input_spec = InputSpec(min_ndim=2)
        self.deg = deg
        self.output_units = output_units
        self.use_bias = use_xbias
        self.activation = activations.get(activation)
        self.kernel_regularizer = tfk.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tfk.constraints.get(kernel_constraint)
        self.init_w = init_w

        if self.init_w is not None and len(self.init_w) != self.deg + 1:
            raise ValueError(
                f"If you specify initial weight for {self.deg}-deg polynomial, "
                f"you must provide {self.deg + 1} weights"
            )

    def build(self, input_shape):
        assert len(input_shape) >= 2

        try:
            self.input_dim = input_shape[-1].value
        except AttributeError:
            self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.deg + 1, self.input_dim, self.output_units),
            initializer="random_normal",
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.init_w is not None:
            for k in range(self.output_units):
                for j in range(self.input_dim):
                    for i in range(self.deg + 1):
                        tfk.backend.set_value(
                            self.kernel[i, j, k], self.init_w[i][j][k]
                        )

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer which is just n-deg P(inputs)
        :rtype: tf.Tensor
        """
        polylist = []
        output_list = []
        for k in range(self.output_units):
            for j in range(self.input_dim):
                polylist.append(
                    [
                        tf.multiply(intpow_avx2(inputs[:, j], i), self.kernel[i, j, k])
                        for i in range(self.deg + 1)
                    ]
                )
                if self.use_bias:
                    polylist[j].append(inputs[:, j])
            output_list.append(
                tf.add_n([tf.add_n(polylist[jj]) for jj in range(self.input_dim)])
            )
        output = tf.stack(output_list, axis=-1)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return tuple((input_shape[0], self.output_units))

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {
            "degree": self.deg,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "initial_weights": self.init_w,
            "kernel_regularizer": tfk.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": tfk.constraints.serialize(self.kernel_constraint),
        }
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}


class TensorInput(Layer):
    """
    TensorInput layer

    :param tensor: tensor, usually is a tensor generating random number
    :type tensor: tf.Tensor
    :return: A layer
    :rtype: object
    :History: 2020-May-3 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, tensor=None, name=None, **kwargs):
        self.supports_masking = True
        self.tensor = tensor
        if not name:
            prefix = self.__class__.__name__
            name = prefix + "_" + str(tfk.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.input_shape

    def call(self, inputs, training=None):
        """
        :Note: Equivalent to __call__()
        :param inputs: Tensor to be applied
        :type inputs: tf.Tensor
        :return: Tensor after applying the layer which is just the masked tensor
        :rtype: tf.Tensor
        """
        return self.tensor

    def get_config(self):
        """
        :return: Dictionary of configuration
        :rtype: dict
        """
        config = {"None": None}
        base_config = super().get_config()
        return {**dict(base_config.items()), **config}


# class BayesPolyFit(DenseVariational_Layer):
#     """
#     | n-deg polynomial fitting layer which acts as a bayesian neural network layer to be optimized with
#     | local reparameterization gradients
#     |
#     | Moreover, the current implementation of this layer does not allow it to be run with Keras,
#     | pleas modify astroNN configure in ~/config.ini key -> tensorflow_keras = tensorflow
#
#     :param deg: degree of polynomial
#     :type deg: int
#     :param output_units: number of output neurons
#     :type output_units: int
#     :param use_xbias: If True, then fitting output=P(inputs)+inputs, else fitting output=P(inputs)
#     :type use_xbias: bool
#     :param init_w: [Optional] list of initial weights if there is any, the list should be [n-degree, input_size, output_size]
#     :type init_w: Union[NoneType, list]
#     :param name: [Optional] name of the layer
#     :type name: Union[NoneType, str]
#     :param activation: [Optional] activation, default is 'linear'
#     :type activation: Union[NoneType, str]
#     :param trainable: [Optional] trainable or not
#     :type trainable: bool
#     :return: A layer
#     :rtype: object
#     :History: 2018-Sept-08 - Written - Henry Leung (University of Toronto)
#     """
#
#     def __init__(self,
#                  deg=1,
#                  output_units=1,
#                  use_xbias=True,
#                  init_w=None,
#                  name=None,
#                  activation=None,
#                  trainable=True,
#                  kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
#                  kernel_posterior_tensor_fn=lambda d: d.sample(),
#                  kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
#                  kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p)):
#         if 'tf' not in keras.__version__:
#             raise EnvironmentError("The current implementation of this layer does not allow it to be run with Keras, "
#                                    "pleas modify astroNN configure in ~/config.ini key -> tensorflow_keras = tensorflow")
#         super().__init__(name=name,
#                          units=output_units,
#                          trainable=trainable,
#                          kernel_posterior_fn=kernel_posterior_fn,
#                          kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
#                          kernel_prior_fn=kernel_prior_fn,
#                          kernel_divergence_fn=kernel_divergence_fn)
#         self.input_spec = InputSpec(min_ndim=2)
#         self.deg = deg
#         self.output_units = output_units
#         self.use_bias = use_xbias
#         self.activation = activations.get(activation)
#         self.kernel_posterior_fn = kernel_posterior_fn
#         self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
#         self.kernel_prior_fn = kernel_prior_fn
#         self.kernel_divergence_fn = kernel_divergence_fn
#         self.init_w = init_w
#
#         if self.init_w is not None and len(self.init_w) != self.deg + 1:
#             raise ValueError(f"If you specify initial weight for {self.deg}-deg polynomial, "
#                              f"you must provide {self.deg+1} weights")
#
#     def _apply_variational_kernel(self, inputs):
#
#         if (not isinstance(self.kernel_posterior, tfd.Independent) or
#                 not isinstance(self.kernel_posterior.distribution, tfd.Normal)):
#             raise TypeError(f'`DenseFlipout` requires kernel_posterior_fn` produce an instance of '
#                             f'`tf.distributions.Independent(tf.distributions.Normal)'
#                             f'`(saw: \"{self.kernel_posterior.name}\").')
#         self.kernel_posterior_affine = tfd.Normal(
#             loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
#             scale=self.kernel_posterior.distribution.scale)
#         self.kernel_posterior_affine_tensor = (self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
#         self.kernel_posterior_tensor = None
#
#         input_shape = tf.shape(inputs)
#         batch_shape = input_shape[:-1]
#
#         sign_input = random_rademacher(input_shape, dtype=inputs.dtype)
#         sign_output = random_rademacher(
#             tf.concat([batch_shape,
#                        tf.expand_dims(self.units, 0)], 0),
#             dtype=inputs.dtype)
#         perturbed_inputs = self._ndegmul(inputs * sign_input,
#                                          self.kernel_posterior_affine_tensor) * sign_output
#
#         outputs = self._ndegmul(inputs, self.kernel_posterior.distribution.loc)
#         outputs += perturbed_inputs
#         return outputs
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_shape = tf.TensorShape(input_shape)
#         in_size = input_shape.with_rank_at_least(2)[-1].value
#
#         if isinstance(input_shape[-1], tf.Dimension):
#             self.input_dim = input_shape[-1].value
#         else:
#             self.input_dim = input_shape[-1]
#
#         self.kernel_posterior = self.kernel_posterior_fn(tf.float32, [self.deg + 1, self.input_dim, self.output_units],
#                                                          'kernel_posterior',
#                                                          self.trainable, self.add_variable)
#         if self.kernel_prior_fn is None:
#             self.kernel_prior = None
#         else:
#             self.kernel_prior = self.kernel_prior_fn(tf.float32, [self.deg + 1, self.input_dim, self.output_units],
#                                                      'kernel_prior', self.trainable, self.add_variable)
#         self._built_kernel_divergence = False
#
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
#         self.built = True
#
#     def _apply_divergence(self, divergence_fn, posterior, prior,
#                           posterior_tensor, name):
#         if divergence_fn is None or posterior is None or prior is None:
#             divergence = None
#             return
#         divergence = tf.identity(divergence_fn(posterior, prior, posterior_tensor), name=name)
#         self.add_loss(divergence)
#
#     def _ndegmul(self, inputs, kernel):
#         polylist = []
#         output_list = []
#         for k in range(self.output_units):
#             for j in range(self.input_dim):
#                 polylist.append([tf.multiply(intpow_avx2(inputs[:, j], i),
#                                              kernel[i, j, k]) for i in range(self.deg + 1)])
#                 if self.use_bias:
#                     polylist[j].append(inputs[:, j])
#             output_list.append(tf.add_n([tf.add_n(polylist[jj]) for jj in range(self.input_dim)]))
#         output = tf.stack(output_list, axis=-1)
#
#         return output
#
#     def call(self, inputs):
#         """
#         :Note: Equivalent to __call__()
#         :param inputs: Tensor to be applied
#         :type inputs: tf.Tensor
#         :return: Tensor after applying the layer which is just n-deg P(inputs)
#         :rtype: tf.Tensor
#         """
#         outputs = self._apply_variational_kernel(inputs)
#
#         if self.activation is not None:
#             outputs = self.activation(outputs)
#
#         if not self._built_kernel_divergence:
#             self._apply_divergence(self.kernel_divergence_fn,
#                                    self.kernel_posterior,
#                                    self.kernel_prior,
#                                    self.kernel_posterior_tensor,
#                                    name='divergence_kernel')
#             self._built_kernel_divergence = True
#         return outputs
#
#     def get_weights_and_error(self):
#         """
#         :return: dictionary contains `weights` for weights and `error` for weights uncertainty
#         :rtype: dict
#         """
#         weights = self.kernel_posterior.distribution.loc.eval(session=keras.backend.get_session())
#         error = self.kernel_posterior.distribution.scale.eval(session=keras.backend.get_session())
#         return {'weights': weights, 'error': error}
#
#     def compute_output_shape(self, input_shape):
#         return tuple((input_shape[0], self.output_units))
#
#     def get_config(self):
#         """
#         :return: Dictionary of configuration
#         :rtype: dict
#         """
#         config = {'degree': self.deg,
#                   'use_bias': self.use_bias,
#                   'activation': activations.serialize(self.activation),
#                   'initial_weights': self.init_w}
#         base_config = super().get_config()
#         return {**dict(base_config.items()), **config}
