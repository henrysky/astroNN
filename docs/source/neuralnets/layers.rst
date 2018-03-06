
Custom Layers in astroNN
=======================================

astroNN provides some customized layers which built on Keras and Tensorflow. Thus they are compatible with Keras
with Tensorflow backend. You can just treat astroNN customized layers as conventional Keras layers.

Dropout Layer for Bayesian Neural Network
---------------------------------------------

`BayesianDropout` is basically Keras's Dropout layer without `seed` and `noise_shape` arguement support. Moreover,
the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.

Dropout can be described by the following formula, lets say we have :math:`i` neurones after activation with value :math:`y_i`

.. math::

   r_{i} = \text{Bernoulli} (p) \\
   \hat{y_i} = r_{i} * y_i


`BayesianDropout` can be imported by

.. code-block:: python

    from astroNN.nn.layers import BayesianDropout

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = BayesianDropout(0.2)(some_keras_layer)
        return model

If you really want to disable the dropout, you do it by

.. code-block:: python

    # Your keras_model define here, assuming you are using functional API
    b_dropout = BayesianDropout(0.2, disable=True)(some_keras_layer)


Dropout w/ Continuous Relaxation for Bayesian Neural Net
---------------------------------------------------------

`ConcreteDropout` is an implementation of `arXiv:1705.07832`_, modified from the original implementation `here`_.
Moreover, the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.
This layer should be only used for experimental purpose only as it has not been tested rigorously.

The main difference between `ConcreteDropout` and standard bernoulli dropout is `ConcreteDropout` learns dropout rate
during training instead of a fixed probability.

`ConcreteDropout` can be imported by

.. code-block:: python

    from astroNN.nn.layers import ConcreteDropout

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = ConcreteDropout()(some_keras_layer)
        return model

If you really want to disable the dropout, you do it by

.. code-block:: python

    # Your keras_model define here, assuming you are using functional API
    b_dropout = ConcreteDropout(disable=True)(some_keras_layer)

.. _arXiv:1705.07832: https://arxiv.org/abs/1705.07832
.. _here: https://github.com/yaringal/ConcreteDropout

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


Time Distributed Layers for Mean and Variance Calculation
----------------------------------------------------------

`TimeDistributedMeanVar` is a layer designed to be used with Bayesian Neural Network with Dropout Variational Inference.
`TimeDistributedMeanVar` should be used with `BayesianRepeatVector` in general.
The advantage of `TimeDistributedMeanVar` layer is you can copy the data and calculate the mean and variance on GPU (if any)
when you are doing dropout variational inference.

`TimeDistributedMeanVar` can be imported by

.. code-block:: python

    from astroNN.nn.layers import TimeDistributedMeanVar

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        monte_carlo_dropout = BayesianRepeatVector(mc_num_here)
        # some layer here, you should use BayesianDropout from astroNN instead of Dropout from Tensorflow:)
        result_mean_var = TimeDistributedMeanVar()(previous_layer_here)
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

`BayesianRepeatVector` is a layer designed to be used with Bayesian Neural Network with Dropout Variational Inference.
`BayesianRepeatVector` should be used with `TimeDistributedMeanVar` in general.
The advantage of `BayesianRepeatVector` layer is you can copy the data and calculate the mean and variance on GPU (if any)
when you are doing dropout variational inference.

`BayesianRepeatVector` will do nothing during training time and repeat vector during testing time as required by
Bayesian Neural Network

`BayesianRepeatVector` can be imported by

.. code-block:: python

    from astroNN.nn.layers import BayesianRepeatVector

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        monte_carlo_dropout = BayesianRepeatVector(mc_num_here)
        # some layer here, you should use BayesianDropout from astroNN instead of Dropout from Tensorflow:)
        result_mean_var = TimeDistributedMeanVar()(previous_layer_here)
        return model

    model.compile(loss=loss_func_here, optimizer=optimizer_here)

    # Use the model to predict
    output = model.predict(x)

    # with dropout variational inference
    # prediction and model uncertainty (variance) from the model
    mean = output[0]
    variance = output[1]
