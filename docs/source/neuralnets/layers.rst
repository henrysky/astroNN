.. automodule:: astroNN.nn.layers

Layers - **astroNN.nn.layers**
===============================

astroNN provides some customized layers which built on Keras and Tensorflow. Thus they are compatible with Keras
with Tensorflow backend. You can just treat astroNN customized layers as conventional Keras layers.

Monte Carlo Dropout Layer
---------------------------------------------

.. autoclass:: astroNN.nn.layers.MCDropout
    :members: call, get_config

`MCDropout` is basically Keras's Dropout layer without `seed` argument support. Moreover,
the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.

Dropout can be described by the following formula, lets say we have :math:`i` neurones after activation with value :math:`y_i`

.. math::

   r_{i} = \text{Bernoulli} (p) \\
   \hat{y_i} = r_{i} * y_i

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = MCDropout(0.2)(some_keras_layer)
        return model

If you really want to disable the dropout, you do it by

.. code-block:: python

    # Your keras_model define here, assuming you are using functional API
    b_dropout = MCDropout(0.2, disable=True)(some_keras_layer)


Monte Carlo Dropout with Continuous Relaxation Layer Wrapper
--------------------------------------------------------------

.. autoclass:: astroNN.nn.layers.MCConcreteDropout
    :members: call, get_config

`MCConcreteDropout` is an implementation of `arXiv:1705.07832`_, modified from the original implementation `here`_.
Moreover, the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.
This layer should be only used for experimental purpose only as it has not been tested rigorously. `MCConcreteDropout` is
technically a layer wrapper instead of a standard layer, so it needs to take a layer as an input argument.

The main difference between `MCConcreteDropout` and standard bernoulli dropout is `MCConcreteDropout` learns dropout rate
during training instead of a fixed probability. Turning/learning dropout rate is not a novel idea, it can be traced back
to one of the original paper `arXiv:1506.02557`_ on variational dropout. But `MCConcreteDropout` focuses on the role
and importance of dropout with Bayesian technique.

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        c_dropout = MCConcreteDropout(some_keras_layer)(previous_layer)
        return model

If you really want to disable the dropout, you do it by

.. code-block:: python

    # Your keras_model define here, assuming you are using functional API
    c_dropout = MCConcreteDropout((some_keras_layer), disable=True)(previous_layer)

.. _arXiv:1705.07832: https://arxiv.org/abs/1705.07832
.. _arXiv:1506.02557: https://arxiv.org/abs/1506.02557
.. _here: https://github.com/yaringal/ConcreteDropout

Monte Carlo Spatial Dropout Layer
--------------------------------------------------

`MCSpatialDropout1D` should be used with Conv1D and `MCSpatialDropout2D` should be used with Conv2D

.. autoclass:: astroNN.nn.layers.MCSpatialDropout1D
    :members: call, get_config

.. autoclass:: astroNN.nn.layers.MCSpatialDropout2D
    :members: call, get_config

`MCSpatialDropout1D` and `MCSpatialDropout2D` are basically Keras's Spatial Dropout layer without
`seed` and `noise_shape` argument support. Moreover, the layers will ignore Keras's learning phase flag,
so the layers will always stays on even in prediction phase.

This version performs the same function as Dropout, however it drops
entire 1D feature maps instead of individual elements. If adjacent frames
within feature maps are strongly correlated (as is normally the case in
early convolution layers) then regular dropout will not regularize the
activations and will otherwise just result in an effective learning rate
decrease. In this case, SpatialDropout1D will help promote independence
between feature maps and should be used instead.

For technical detail, you can refer to the original paper `arXiv:1411.4280`_

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = MCSpatialDropout1D(0.2)(keras_conv_layer)
        return model

If you really want to disable the dropout, you do it by

.. code-block:: python

    # Your keras_model define here, assuming you are using functional API
    b_dropout = MCSpatialDropout1D(0.2, disable=True)(keras_conv_layer)


.. _arXiv:1411.4280: https://arxiv.org/abs/1411.4280

Monte Carlo Gaussian Dropout Layer
---------------------------------------------

.. autoclass:: astroNN.nn.layers.MCGaussianDropout
    :members: call, get_config

`MCGaussianDropout` is basically Keras's Dropout layer without `seed` argument support. Moreover,
the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.

`MCGaussianDropout` should be used with caution for Bayesian Neural Network: https://arxiv.org/abs/1711.02989

Gaussian Dropout can be described by the following formula, lets say we have :math:`i` neurones after activation with value :math:`y_i`

.. math::

   r_{i} = \mathcal{N}\bigg(1, \sqrt{\frac{p}{1-p}}\bigg) \\
   \hat{y_i} = r_{i} * y_i

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = MCGaussianDropout(0.2)(some_keras_layer)
        return model

If you really want to disable the dropout, you do it by

.. code-block:: python

    # Your keras_model define here, assuming you are using functional API
    b_dropout = MCGaussianDropout(0.2, disable=True)(some_keras_layer)

Monte Carlo Batch Normalization Layer
---------------------------------------------

.. autoclass:: astroNN.nn.layers.MCBatchNorm
    :members: call, get_config

`MCBatchNorm` is a layer doing Batch Normalization originally described in arViX: https://arxiv.org/abs/1502.03167

`MCBatchNorm` should be used with caution for Bayesian Neural Network: https://openreview.net/forum?id=BJlrSmbAZ

Batch Normalization can be described by the following formula, lets say we have :math:`N` neurones after activation for a layer

.. math::

   N_{i} = \frac{N_{i} - \text{Mean}[N]}{\sqrt{\text{Var}[N]}}


`MCBatchNorm` can be imported by

.. code-block:: python

    from astroNN.nn.layers import MCBatchNorm

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = MCBatchNorm()(some_keras_layer)
        return model


Error Propagation Layer
---------------------------------------------

`ErrorProp` is a layer designed to do error propagation in neural network. It will acts as an identity transformation
layer during training phase but add gaussian noise to input during test phase. The idea is if you have known uncertainty
in input, and you want to understand how input uncertainty (more specifically this layer assuming the uncertainty is
Gaussian) affects the output. Since this layer add random known Gaussian uncertainty to the input, you can run model
prediction a few times to get some predictions, mean of those predictions will be the final prediction and standard
derivation of the predictions will be the propagated uncertainty.


`ErrorProp` can be imported by

.. code-block:: python

    from astroNN.nn.layers import ErrorProp

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        input_with_error = ErrorProp(some_gaussian_tensor)(input)
        return model


KL-Divergence Layer for Variational Autoencoder
-------------------------------------------------

`KLDivergenceLayer` is a layer designed to be used in Variational Autoencoder. It will acts as an identity transformation
layer but will add KL-divergence to the total loss.

`KLDivergenceLayer` can be imported by

.. code-block:: python

    from astroNN.nn.layers import KLDivergenceLayer

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        z_mu = Encoder_Mean_Layer(.....)
        z_log_var = Encoder_Var_Layer(.....)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        # And then decoder or whatever
        return model


Mean and Variance Calculation Layer for Bayesian Neural Net
------------------------------------------------------------

`FastMCInferenceMeanVar` is a layer designed to be used with Bayesian Neural Network with Dropout Variational Inference.
`FastMCInferenceMeanVar` should be used with `FastMCInference` in general.
The advantage of `FastMCInferenceMeanVar` layer is you can copy the data and calculate the mean and variance on GPU (if any)
when you are doing dropout variational inference.

`FastMCInferenceMeanVar` can be imported by

.. code-block:: python

    from astroNN.nn.layers import FastMCInferenceMeanVar

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        monte_carlo_dropout = FastMCInference(mc_num_here)
        # some layer here, you should use MCDropout from astroNN instead of Dropout from Tensorflow:)
        result_mean_var = FastMCInferenceMeanVar()(previous_layer_here)
        return model

    model.compile(loss=loss_func_here, optimizer=optimizer_here)

    # Use the model to predict
    output = model.predict(x)

    # with dropout variational inference
    # prediction and model uncertainty (variance) from the model
    mean = output[0]
    variance = output[1]

Repeat Vector Layer for Bayesian Neural Net
---------------------------------------------

`FastMCRepeat` is a layer to repeat training data to do Monte Carlo integration required by Bayesian Neural Network.

`FastMCRepeat` is a layer designed to be used with Bayesian Neural Network with Dropout Variational Inference.
`FastMCRepeat` should be used with `FastMCInferenceMeanVar` in general.
The advantage of `FastMCRepeat` layer is you can copy the data and calculate the mean and variance on GPU (if any)
when you are doing dropout variational inference.

`FastMCRepeat` can be imported by

.. code-block:: python

    from astroNN.nn.layers import FastMCRepeat

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        monte_carlo_dropout = FastMCRepeat(mc_num_here)
        # some layer here, you should use MCDropout from astroNN instead of Dropout from Tensorflow:)
        result_mean_var = FastMCInferenceMeanVar()(previous_layer_here)
        return model

    model.compile(loss=loss_func_here, optimizer=optimizer_here)

    # Use the model to predict
    output = model.predict(x)

    # with dropout variational inference
    # prediction and model uncertainty (variance) from the model
    mean = output[0]
    variance = output[1]


Fast Monte Carlo Integration Layer for Keras Model
---------------------------------------------------

.. autoclass:: astroNN.nn.layers.FastMCInference
    :members: __call__, get_config

`FastMCInference` is a layer designed for fast Monte Carlo Inference on GPU. One of the main challenge of MC integration
on GPU is you want the data stay on GPU and you do MC integration on GPU entirely, moving data from drives to GPU is
a very expensive operation. `FastMCInference` will create a new keras model such that it will replicate data on GPU, do
Monte Carlo integration and calculate mean and variance on GPU, and get back the result.

Benchmark (Nvidia GTX1060 6GB): 98,000 7514 pixles APOGEE Spectra, traditionally the 25 forward pass spent ~270 seconds,
by using `FastMCInference`, it only spent ~65 seconds to do the exact same task.

It can only be used with Keras model. If you are using customised model purely with Tensorflow, you should use `FastMCRepeat`
and `FastMCInferenceMeanVar`

You can import the function from astroNN by

.. code-block:: python

    from astroNN.nn.layers import FastMCInference

    # keras_model is your keras model with 1 output which is a concatenation of labels prediction and predictive variance
    keras_model = Model(....)

    # fast_mc_model is the new keras model capable to do fast monte carlo integration on GPU
    fast_mc_model = FastMCInference(keras_model)

    # You can just use keras API with the new model such as
    result = fast_mc_model.predict(.....)

    # here is the result dimension
    predictions = result[:, :(result.shape[1] // 2), 0]  # mean prediction
    mc_dropout_uncertainty = result[:, :(result.shape[1] // 2), 1] * (self.labels_std ** 2)  # model uncertainty
    predictions_var = np.exp(result[:, (result.shape[1] // 2):, 0]) * (self.labels_std ** 2)  # predictive uncertainty
