
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
