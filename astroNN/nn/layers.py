import math
import keras


epsilon = keras.backend.epsilon
initializers = keras.initializers
activations = keras.activations
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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
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
        kl_batch = -0.5 * keras.backend.numpy.sum(
            1 + log_var - keras.backend.numpy.square(mu) - keras.backend.numpy.exp(log_var), axis=-1
        )
        self.add_loss(keras.backend.numpy.mean(kl_batch), inputs=inputs)

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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random.normal(shape=(batch, dim))
        return z_mean + keras.backend.numpy.exp(0.5 * z_log_var) * epsilon


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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = keras.backend.shape(inputs)
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
            return keras.backend.random.dropout(inputs, rate=self.rate, noise_shape=noise_shape)

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
        input_shape = keras.backend.shape(inputs)
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
        input_shape = keras.backend.shape(inputs)
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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
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
            return inputs * keras.backend.random.normal(
                shape=keras.backend.shape(inputs), mean=1.0, stddev=stddev
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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
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
            training = keras.backend.learning_phase()

        noised = keras.backend.random.normal([1], mean=inputs[0], stddev=inputs[1])
        output_tensor = keras.backend.numpy.where(keras.backend.numpy.equal(training, True), inputs[0], noised)
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
        if isinstance(model, keras.Model) or isinstance(model, keras.Sequential):
            self.model = model
        else:
            raise TypeError(
                f"FastMCInference expects tensorflow.keras Model, you gave {type(model)}"
            )
        new_input = keras.layers.Input(shape=(self.model.input_shape[1:]), name="input")
        mc_model = keras.models.Model(
            inputs=self.model.inputs, outputs=self.model.outputs
        )

        mc = FastMCInferenceMeanVar()(
            FastMCInferenceV2_internal(mc_model, self.n)(new_input)
        )
        new_mc_model = keras.models.Model(inputs=new_input, outputs=mc)

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
        if isinstance(model, keras.Model) or isinstance(model, keras.Sequential):
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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
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
        mean, var = keras.backend.nn.moments(inputs, axes=0)
        return keras.backend.numpy.stack((keras.backend.numpy.squeeze([mean]), keras.backend.numpy.squeeze([var])), axis=-1)


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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
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
        expanded_inputs = keras.backend.numpy.expand_dims(inputs, 1)
        # we want [1, self.n, 1.....]
        return keras.backend.numpy.tile(
            expanded_inputs,
            keras.backend.numpy.concat(
                [[1, self.n], keras.backend.numpy.ones_like(keras.backend.numpy.shape(expanded_inputs))[2:]], axis=0
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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
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
                training = keras.backend.learning_phase()
            output_tensor = keras.backend.numpy.where(
                keras.backend.numpy.equal(training, True), tf.stop_gradient(inputs), inputs
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
            name = prefix + "_" + str(keras.backend.get_uid(prefix))
        super().__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = keras.backend.shape(input_shape)
        # TODO: convert to keras
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
        batchsize = keras.backend.shape(inputs)[0]
        # need to reshape because tf.keras cannot get the Tensor shape correctly from tf.boolean_mask op

        boolean_mask = keras.backend.numpy.any(keras.backend.numpy.not_equal(inputs, self.boolmask), axis=1, keepdims=True)

        return keras.backend.numpy.reshape(
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
