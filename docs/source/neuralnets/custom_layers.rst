
Custom Layers in astroNN
=======================================

astroNN provides some customized layers which built on Kera and Tensorflow. Thus they are compatiable with Keras
with Tensorflow backend

Dropout Layer for Bayesian Neural Network
---------------------------------------------

`BayesianDropout` is basically Keras's Dropout layer without `seed` and `noise_shape` arguement support. Moreover,
the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.

`BayesianDropout` can be imported by

.. code-block:: python

    from astroNN.nn.utilities.custom_layers import BayesianDropout

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = BayesianDropout(0.2)(some_keras_layer)
        return model


Error Propagation Layer
---------------------------------------------

`ErrorProp` is a layer designed to do error propagation in neural network. It will acts as an identity transformation
layer during training phase and add gaussian noise to input during test phase. The idea is if you have known uncertainty
in input, and you want to understand how input uncertainty (more specifically this layer assuming the uncertainty is
Gaussian) affects the output. Since this layer add random known Gaussian uncertainty to the input, you can run model
prediction a few times to get some prediction, mean of the predictions will be the final prediction and standard
derivation of the predictions will be the propagated uncertainty.


`ErrorProp` can be imported by

.. code-block:: python

    from astroNN.nn.utilities.custom_layers import ErrorProp

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        input = Input(.....)
        input_with_error = ErrorProp(some_gaussian_tensor)(input)
        return model