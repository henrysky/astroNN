
Custom Layers in astroNN
=======================================

astroNN provides some customized layers which built on Kera and Tensorflow. Thus they are compatiable with Keras
with Tensorflow backend

Dropout Layer for Bayesian Neural Network
---------------------------------------------

`BayesianDropout` is basically Keras's Dropout layer without `seed` and `noise_shape` arguement support. Moreover,
the layer will ignore Keras's learning phase flag, so the layer will always stays on even in prediction phase.

Dropout Layer for Bayesian Neural Network can be imported by

.. code:: python

    from astroNN.nn import BayesianDropout

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here, assuming you are using functional API
        b_dropout = BayesianDropout(0.2)(some_keras_layer)
        return model