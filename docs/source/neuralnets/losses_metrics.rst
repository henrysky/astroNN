
Custom Loss Functions and Metrics in astroNN
==============================================

astroNN provides modified loss functions which are capable to deal with missing labels which are represented by ``magicnumber``
in astroNN configuration file or ``Magic Number`` in equations below.
Since they are built on Tensorflow and follows Keras API requirement, all astroNN loss functions are fully compatible
with Keras with Tensorflow backend, as well as directly be imported and used with Tensorflow, for most loss functions, the
first argument is ground truth tensor and the second argument is prediction tensor from neural network.

.. note:: Always make sure when you are normalizing your data, keep the magic number as magic number. If you use astroNN normalizer, astroNN will take care of that.

Here are some explanations on variables in the following loss functions:

:math:`y_i` means the ground truth labels, or target labels

:math:`\hat{y_i}` means the prediction from neural network

Correction Term for Magic Number
----------------------------------

Since astroNN deals with magic number by assuming the prediction from neural network for those ground truth with Magic Number
is right, so we need a correction term.

The correction term in astroNN is defined by the following equation and we call the equation :math:`\mathcal{F}_{correction}`

.. math::

   \mathcal{F}_{correction} = \frac{\text{Non-Magic Number Count} + \text{Magic Number Count}}{\text{Non Magic Number Count}}

In case of no labels with Magic Number is presented, :math:`\mathcal{F}_{correction}` will equal to 1

Mean Squared Error
-----------------------

MSE is based on the equation

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            (\hat{y_i}-y_i)^2 & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})


MSE can be imported by

.. code-block:: python

    from astroNN.nn.losses import mean_squared_error

    # OR it can be imported by
    from astroNN.nn.metrics import mean_squared_error

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=mean_squared_error, ...)

Mean Absolute Error
-----------------------

MAE is based on the equation

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            \left| \hat{y_i}-y_i \right| & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})


MAE can be imported by

.. code-block:: python

    from astroNN.nn.losses import mean_absolute_error

    # OR it can be imported by
    from astroNN.nn.metrics import mean_absolute_error

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=mean_absolute_error, ...)


Regression Loss and Predictive Variance Loss for Bayesian Neural Net
------------------------------------------------------------------------

It is based on the equation, please notice :math:`s_i` is  representing
:math:`log((\sigma_{predictive, i})^2 + (\sigma_{known, i})^2)`. Neural network not predicting variance
directly to avoid numerical instability but predicting :math:`log((\sigma_{i})^2)`

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            \frac{1}{2} (\hat{y_i}-y_i)^2 e^{-s_i} + \frac{1}{2}(s_i) & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{BNN} = \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})

Regression Loss for Bayesian Neural Net can be imported by

.. code-block:: python

    from astroNN.nn.losses import mse_lin_wrapper, mse_var_wrapper

`mse_lin_wrapper` is for the prediction neurones

`mse_var_wrapper` is for the predictive variance neurones

They basically do the same things and can be used with Keras, you just have to import the functions from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here

        # model for the training process
        model = Model(inputs=[input_tensor, labels_err_tensor], outputs=[output, predictive_variance])

        # model for the prediction
        model_prediction = Model(inputs=input_tensor, outputs=[output, variance_output])

        predictive_variance = Dense(name='predictive_variance', ...)
        output = Dense(name='output', ...)

        predictive_variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(predictive_variance, labels_err_tensor)

        return model, model_prediction, output_loss, predictive_variance_loss

    model, model_prediction, output_loss, predictive_variance_loss = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss={'output': output_loss, 'predictive_variance': predictive_variance_loss}, ...)

.. note:: If you don't have the known labels uncertainty, you can just give an array of zeros as your labels uncertainty

Mean Squared Logarithmic Error
--------------------------------

MSLE  will first clip the values of prediction from neural net for the sake of numerical stability,

.. math::

   y_i = \begin{cases}
        \begin{split}
            \epsilon + 1 & \text{ for } y_i < \epsilon \\
            y_i + 1 & \text{ for otherwise }
        \end{split}
    \end{cases}

   \text{where } \epsilon \text{ is a small constant}

Then MSLE is based on the equation

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            (\log{(\hat{y_i})} - \log{(y_i)})^2 & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})


MSLE can be imported by

.. code-block:: python

    from astroNN.nn.losses import mean_absolute_percentage_error

    # OR it can be imported by
    from astroNN.nn.metrics import mean_absolute_percentage_error

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=mean_absolute_percentage_error, ...)

Mean Absolute Percentage Error
--------------------------------

Mean Absolute Percentage Error will first clip the values of prediction from neural net for the sake of numerical stability,

.. math::

   y_i = \begin{cases}
        \begin{split}
            \epsilon  & \text{ for } y_i < \epsilon \\
            y_i & \text{ for otherwise }
        \end{split}
    \end{cases}

   \text{where } \epsilon \text{ is a small constant}

Then Mean Absolute Percentage Error is based on the equation

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            100 \text{ } \frac{\left| y_i - \hat{y_i} \right|}{y_i} & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})


Mean Absolute Percentage Error can be imported by

.. code-block:: python

    from astroNN.nn.losses import mean_absolute_percentage_error

    # OR it can be imported by
    from astroNN.nn.metrics import mean_absolute_percentage_error

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=mean_absolute_percentage_error, ...)

Categorical Cross-Entropy
----------------------------

Categorical Cross-Entropy will first clip the values of prediction from neural net for the sake of numerical stability if
the prediction is not coming from logits (before softmax activated)

.. math::

   \hat{y_i} = \begin{cases}
        \begin{split}
            \epsilon & \text{ for } \hat{y_i} < \epsilon \\
            1 - \epsilon & \text{ for } \hat{y_i} > 1 - \epsilon \\
            \hat{y_i} & \text{ for otherwise }
        \end{split}
    \end{cases}

   \text{where } \epsilon \text{ is a small constant}

and then based on the equation

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            y_i \log{(\hat{y_i})} & \text{ for } y_i \neq \text{Magic Number}\\
            \hat{y_i} \log{(\hat{y_i})} & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = - \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})

Categorical Cross-Entropy can be imported by

.. code-block:: python

    from astroNN.nn.losses import categorical_cross_entropy

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=categorical_cross_entropy(from_logits=False), ...)

.. note:: astroNN's categorical_cross_entropy expects values after softmax activated by default. If you want to use logits, please use from_logits=True

Binary Cross-Entropy
----------------------------

Binary Cross-Entropy will first clip the values of prediction from neural net for the sake of numerical stability if
the prediction is not coming from logits (before softmax activated)

.. math::

   \hat{y_i} = \begin{cases}
        \begin{split}
            \epsilon & \text{ for } \hat{y_i} < \epsilon \\
            1 - \epsilon & \text{ for } \hat{y_i} > 1 - \epsilon \\
            \hat{y_i} & \text{ for otherwise }
        \end{split}
    \end{cases}

   \text{where } \epsilon \text{ is a small constant}

and then based on the equation

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            y_i \log{(\hat{y_i})} + (1-y_i)\log{(1-\hat{y_i})} & \text{ for } y_i \neq \text{Magic Number}\\
            \hat{y_i} \log{(\hat{y_i})} + (1-\hat{y_i})\log{(1-\hat{y_i})} & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = - \frac{1}{D} \sum_{i=1}^{batch} (Loss_i \mathcal{F}_{correction, i})

Binary Cross-Entropy can be imported by

.. code-block:: python

    from astroNN.nn.losses import binary_cross_entropy

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=binary_cross_entropy(from_logits=False), ...)

.. note:: astroNN's binary_cross_entropy expects values after softmax activated by default. If you want to use logits, please use from_logits=True

Categorical Classification Accuracy
------------------------------------

Categorical Classification Accuracy will first deal with Magic Number

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            y_i & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

Then based on the equation

.. math::

   Accuracy_i = \begin{cases}
        \begin{split}
          1 & \text{ for } \text{Argmax}(y_i) = \text{Argmax}(\hat{y_i})\\
          0 & \text{ for } \text{Argmax}(y_i) \neq \text{Argmax}(\hat{y_i})
        \end{split}
    \end{cases}

And thus the accuracy for is

.. math::

   Accuracy = \frac{1}{D} \sum_{i=1}^{labels} (Accuracy_i \mathcal{F}_{correction, i})

Categorical Classification Accuracy can be imported by

.. code-block:: python

    from astroNN.nn.metrics import categorical_accuracy

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's metrics function first
    model.compile(metrics=categorical_accuracy, ...)

.. note:: Please make sure you use categorical_accuracy when using categorical_cross_entropy as the loss function

Binary Classification Accuracy
---------------------------------

Binary Classification Accuracy will round the values of prediction

.. math::

   \hat{y_i} = \begin{cases}
        \begin{split}
            1 & \text{ for } \hat{y_i} > 0.5 \\
            0 & \text{ for } \hat{y_i} \leq 0.5
        \end{split}
    \end{cases}

and then based on the equation

.. math::

   Accuracy_i = \begin{cases}
        \begin{split}
          1 & \text{ for } y_i = \hat{y_i}\\
          0 & \text{ for } y_i \neq \hat{y_i}
        \end{split}
    \end{cases}

And thus the accuracy for is

.. math::

   Accuracy = \frac{1}{D} \sum_{i=1}^{labels} (Accuracy_i \mathcal{F}_{correction, i})

Binary Classification Accuracy can be imported by

.. code-block:: python

    from astroNN.nn.metrics import binary_accuracy

It can be used with Keras, you just have to import the function from astroNN

.. code-block:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's metrics function first
    model.compile(metrics=binary_accuracy, ...)

.. note:: Please make sure you use binary_accuracy when using binary_cross_entropy as the loss function
