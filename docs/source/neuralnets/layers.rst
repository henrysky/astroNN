.. automodule:: astroNN.nn.layers

Layers - **astroNN.nn.layers**
===============================

astroNN provides some customized layers which built on tensorflow.keras. You can just treat astroNN customized layers as conventional Keras layers.

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

And here is an example of usage

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

And here is an example of usage

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

And here is an example of usage

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

And here is an example of usage

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

And here is an example of usage

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = MCBatchNorm()(some_keras_layer)
        return model


Error Propagation Layer
---------------------------------------------

.. autoclass:: astroNN.nn.layers.ErrorProp
    :members: call, get_config


`ErrorProp` is a layer designed to do error propagation in neural network. It will acts as an identity transformation
layer during training phase but add gaussian noise to input during test phase. The idea is if you have known uncertainty
in input, and you want to understand how input uncertainty (more specifically this layer assuming the uncertainty is
Gaussian) affects the output. Since this layer add random known Gaussian uncertainty to the input, you can run model
prediction a few times to get some predictions, mean of those predictions will be the final prediction and standard
derivation of the predictions will be the propagated uncertainty.


`ErrorProp` can be imported by

.. code-block:: python

    from astroNN.nn.layers import ErrorProp

And here is an example of usage

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        input_with_error = ErrorProp()([input, input_error])
        return model


KL-Divergence Layer for Variational Autoencoder
-------------------------------------------------

.. autoclass:: astroNN.nn.layers.KLDivergenceLayer
    :members: call, get_config

`KLDivergenceLayer` is a layer designed to be used in Variational Autoencoder. It will acts as an identity transformation
layer but will add KL-divergence to the total loss.

`KLDivergenceLayer` can be imported by

.. code-block:: python

    from astroNN.nn.layers import KLDivergenceLayer

And here is an example of usage

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        z_mu = Encoder_Mean_Layer(.....)
        z_log_var = Encoder_Var_Layer(.....)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        # And then decoder or whatever
        return model
        

Polynomial Fitting Layer
----------------------------

.. autoclass:: astroNN.nn.layers.PolyFit
    :members: call, get_config

`PolyFit` is a layer designed to do n-degree polynomial fitting in a neural network style by treating coefficient as
neural network weights and optimize them by neural network optimizer. The fitted polynomial(s) are
in the following form (you can specify initial weights by init_w=[[[:math:`w_0`]], [[:math:`w_1`]], ..., [[:math:`w_n`]]]) for a single input and output value

.. math::

    p(x) = w_0 + w_1 * x + ... + w_n * x^n

For multiple i input values and j output values and n-deg polynomial (you can specify initial weights by
init_w=[[[:math:`w_{0, 1, 0}`, :math:`w_{0, 1, 1}`, ..., :math:`w_{0, 1, j}`],
[:math:`w_{0, 2, 0}`, :math:`w_{0, 2, 1}`, ..., :math:`w_{0, 2, j}`], ...
[:math:`w_{0, i, 0}`, :math:`w_{0, i, 1}`, ..., :math:`w_{0, i, j}`]], ...,
[[:math:`w_{n, 1, 0}`, :math:`w_{n, 1, 1}`, ..., :math:`w_{n, 1, j}`],
[:math:`w_{n, 2, 0}`, :math:`w_{n, 2, 1}`, ..., :math:`w_{n, 2, j}`], ...
[:math:`w_{n, i, 0}`, :math:`w_{n, i, 1}`, ..., :math:`w_{n, i, j}`]]])

and the polynomial is as the following form for For multiple i input values and j output values and n-deg polynomial

.. math::

    \text{output neurons from 1 to j} = \begin{cases}
        \begin{split}
            p_1(x) = \sum\limits_{i=1}^i \Big(w_{0, 1, 0} + w_{1, 1, 1} * x_1 + ... + w_{n, 1, i} * x_i^n \Big) \\
            p_2(x) = \sum\limits_{i=1}^i \Big(w_{0, 2, 0} + w_{1, 2, 1} * x_1 + ... + w_{n, 2, i} * x_i^n \Big) \\
            p_{...}(x) = \sum\limits_{i=1}^i \Big(\text{......}\Big) \\
            p_j(x) = \sum\limits_{i=1}^i \Big(w_{0, j, 0} + w_{1, j, 1} * x_1 + ... + w_{n, j, i} * x_i^n \Big) \\
        \end{split}
    \end{cases}

`PolyFit` can be imported by

.. code-block:: python

    from astroNN.nn.layers import PolyFit

And here is an example of usage

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        output = PolyFit(deg=1)(input)
        return model(inputs=input, outputs=output)


To show it works as a polynomial, you can refer the following example:

.. code-block:: python

    import numpy as np
    from astroNN.nn.layers import PolyFit

    from astroNN.shared.nn_tools import cpu_fallback
    import tensorflow.keras as keras

    cpu_fallback()  # force tf to use CPU

    Input = keras.layers.Input
    Model = keras.models.Model

    # Data preparation
    polynomial_coefficient = [0.1, -0.05]
    random_xdata = np.random.normal(0, 3, (100, 1))
    random_ydata = polynomial_coefficient[1] * random_xdata + polynomial_coefficient[0]

    input = Input(shape=[1, ])
    # set initial weights
    output = PolyFit(deg=1, use_xbias=False, init_w=[[[0.1]], [[-0.05]]], name='polyfit')(input)
    model = Model(inputs=input, outputs=output)

    # predict without training (i.e. without gradient updates)
    np.allclose(model.predict(random_xdata), random_ydata)
    >>> True # True means prediction approx close enough


Mean and Variance Calculation Layer for Bayesian Neural Net
------------------------------------------------------------

.. autoclass:: astroNN.nn.layers.FastMCInferenceMeanVar
    :members: call, get_config

If you wnat fast MC inference on GPU and you are using keras models, you should just use FastMCInference_.

`FastMCInferenceMeanVar` is a layer designed to be used with Bayesian Neural Network with Dropout Variational Inference.
`FastMCInferenceMeanVar` should be used with `FastMCInference` in general.
The advantage of `FastMCInferenceMeanVar` layer is you can copy the data and calculate the mean and variance on GPU (if any)
when you are doing dropout variational inference.

`FastMCInferenceMeanVar` can be imported by

.. code-block:: python

    from astroNN.nn.layers import FastMCInferenceMeanVar

And here is an example of usage

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

.. autoclass:: astroNN.nn.layers.FastMCRepeat
    :members: call, get_config

If you wnat fast MC inference on GPU and you are using keras models, you should just use FastMCInference_.

`FastMCRepeat` is a layer to repeat training data to do Monte Carlo integration required by Bayesian Neural Network.

`FastMCRepeat` is a layer designed to be used with Bayesian Neural Network with Dropout Variational Inference.
`FastMCRepeat` should be used with `FastMCInferenceMeanVar` in general.
The advantage of `FastMCRepeat` layer is you can copy the data and calculate the mean and variance on GPU (if any)
when you are doing dropout variational inference.

`FastMCRepeat` can be imported by

.. code-block:: python

    from astroNN.nn.layers import FastMCRepeat

And here is an example of usage

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

.. _FastMCInference:

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

Gradient Stopping Layer
---------------------------------------------

.. autoclass:: astroNN.nn.layers.StopGrad
    :members: call, get_config


It uses ``tf.stop_gradient`` and acts as a Keras layer.

`StopGrad` can be imported by

.. code-block:: python

    from astroNN.nn.layers import StopGrad

It can be used with keras or tensorflow.keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        # some layers ...
        stopped_grad_layer = StopGrad()(...)
        # some layers ...
        return model

For example, if you have a model with multiple branches and you only want error backpropagate to one but not the other,

.. code-block:: python

    from astroNN.nn.layers import StopGrad
    # we use zeros loss just to demonstrate StopGrad works and no error backprop from StopGrad layer
    from astroNN.nn.losses import zeros_loss
    import numpy as np
    from astroNN.shared.nn_tools import cpu_fallback
    import tensorflow.keras as keras

    cpu_fallback()  # force tf to use CPU

    Input = keras.layers.Input
    Dense = keras.layers.Dense
    concatenate = keras.layers.concatenate
    Model = keras.models.Model

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))
    input2 = Input(shape=[7514])
    dense1 = Dense(100, name='normaldense')(input2)
    dense2 = Dense(25, name='wanted_dense')(input2)
    dense2_stopped = StopGrad(name='stopgrad', always_on=True)(dense2)
    output2 = Dense(25, name='wanted_dense2')(concatenate([dense1, dense2_stopped]))
    model2 = Model(inputs=input2, outputs=[output2, dense2])
    model2.compile(optimizer=keras.optimizers.SGD(lr=0.1),
                   loss={'wanted_dense2': 'mse', 'wanted_dense': zeros_loss})
    weight_b4_train = model2.get_layer(name='wanted_dense').get_weights()[0]
    weight_b4_train2 = model2.get_layer(name='normaldense').get_weights()[0]
    model2.fit(random_xdata, [random_ydata, random_ydata])
    weight_a4_train = model2.get_layer(name='wanted_dense').get_weights()[0]
    weight_a4_train2 = model2.get_layer(name='normaldense').get_weights()[0]

    print(np.all(weight_b4_train == weight_a4_train))
    >>> True  # meaning all the elements from Dense with StopGrad layer are equal due to no gradient update
    print(np.all(weight_b4_train2 == weight_a4_train2))
    >>> False  # meaning not all the elements from normal Dense layer are equal due to gradient update


Boolean Masking Layer
-----------------------

.. autoclass:: astroNN.nn.layers.BoolMask
    :members: call, get_config


`BoolMask` takes numpy boolean array as layer initialization and mask the input tensor.

`BoolMask` can be imported by

.. code-block:: python

    from astroNN.nn.layers import BoolMask

It can be used with keras or tensorflow.keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        # some layers ...
        stopped_grad_layer = BoolMask(mask=....)(...)
        # some layers ...
        return model
