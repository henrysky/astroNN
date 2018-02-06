
Custom Layers in astroNN
=======================================

astroNN provides some customized layers which built on Kera and Tensorflow. Thus they are compatiable with Keras
with Tensorflow backend

Dropout Layer for Bayesian Neural Network
---------------------------------------------

Dropout Layer for Bayesian Neural Network is basically Keras's Dropout without `seed` and `noise_shape` support. Moreover,
`BayesianDropout` layer will ignore Keras's learning phase flag, so the layer will always stays on.

Dropout Layer for Bayesian Neural Network can be imported by

.. code:: python

    from astroNN.nn import BayesianDropout

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here
        some_keras_layer = keras_layer(x)(y)
        b_dropout = BayesianDropout(0.2)(some_keras_layer)
        some_keras_layer = keras_layer(x)(b_dropout)
        return model