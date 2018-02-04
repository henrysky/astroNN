
Loss and metrics functions in astroNN
=======================================

astroNN provides modified loss function which is capable to deal with missing labels (represented by Magic Number).
Since they are similar to Keras and built on Tensorflow, all astroNN loss functions are fully compatible with Keras with
Tensorflow backend.

.. note:: Always make sure when you are normalizing your data, keep the magic number as magic number. If you use astroNN normalizer, astroNN will take care of that.

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

   Loss_{NN} = \frac{1}{D} \sum_{i=1}^{batch} Loss_i


MSE can be imported by

.. code:: python

    from astroNN.models.losses import mean_squared_error

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=mean_squared_error, ...)

Mean Abolute Error
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

   Loss_{NN} = \frac{1}{D} \sum_{i=1}^{batch} Loss_i


MAE can be imported by

.. code:: python

    from astroNN.models.losses import mean_absolute_error

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=mean_absolute_error, ...)


Regression Loss and Predictive Variance Loss for Bayesian Neural Net
------------------------------------------------------------------------

It is based on the equation, please notice :math:`s_i` is :math:`log((\sigma_{predictive, i})^2 + (\sigma_{known, i})^2)`
to avoid numerical instability

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            \frac{1}{2} (\hat{y_i}-y_i)^2 e^{-s_i} + \frac{1}{2}(s_i) & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{BNN} = \frac{1}{D} \sum_{i=1}^{batch} Loss_i

Regression Loss for Bayesian Neural Net can be imported by

.. code:: python

    from astroNN.models.losses import mse_lin_wrapper, mse_var_wrapper

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here

        # model for the training process
        model = Model(inputs=[input_tensor, labels_err_tensor], outputs=[output, predictive_variance])

        # model for the prediction
        model_prediction = Model(inputs=input_tensor, outputs=[output, variance_output])

        predictive_variance = ...(name='predictive_variance', ...)
        output = ...(name='output', ...)

        predictive_variance_loss = mse_var_wrapper(output, labels_err_tensor)
        output_loss = mse_lin_wrapper(predictive_variance, labels_err_tensor)

        return model, model_prediction, output_loss, predictive_variance_loss

    model, model_prediction, output_loss, predictive_variance_loss = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss={'output': output_loss, 'predictive_variance': predictive_variance_loss}, ...)

.. note:: If you don't have the knwon labels variance, you can just supply an array of zero as your labels variance and let BNN deals with predictive variance only

Categorical Cross-Entropy
----------------------------

Categorical Cross-Entropy will first clip the values of prediction from neural net for the sake of numerical stability

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
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = - \frac{1}{D} \sum_{i=1}^{batch} Loss_i

Categorical Cross-Entropy can be imported by

.. code:: python

    from astroNN.models.losses import categorical_cross_entropy

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's loss function first
    model.compile(loss=categorical_cross_entropy(from_logits=False), ...)

.. note:: astroNN's categorical_cross_entropy expects values after softmax activated by default. If you want to use logits, please use from_logits=True

Binary Cross-Entropy
----------------------------

Binary Cross-Entropy will first clip the values of prediction from neural net for the sake of numerical stability

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
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{NN} = - \frac{1}{D} \sum_{i=1}^{batch} Loss_i

Categorical Cross-Entropy can be imported by

.. code:: python

    from astroNN.models.losses import binary_cross_entropy

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

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

   Accuracy = \frac{1}{D} \sum_{i=1}^{labels} Accuracy_i

Categorical Classification Accuracy can be imported by

.. code:: python

    from astroNN.models.utilities.metrics import categorical_accuracy

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's metrics function first
    model.compile(metrics=categorical_accuracy, ...)

.. note:: make sure you use categorical_accuracy when using categorical_cross_entropy as the loss function

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
          1 & \text{ for } y_i = \hat{y+_i}\\
          0 & \text{ for } y_i \neq \hat{y+_i}
        \end{split}
    \end{cases}

And thus the accuracy for is

.. math::

   Accuracy = \frac{1}{D} \sum_{i=1}^{labels} Accuracy_i

Binary Classification Accuracy can be imported by

.. code:: python

    from astroNN.models.utilities.metrics import binary_accuracy

It can be used with Keras, you just have to import the function from astroNN

.. code:: python

    def keras_model():
        # Your keras_model define here
        return model

    model = keras_model()
    # remember to import astroNN's metrics function first
    model.compile(metrics=binary_accuracy, ...)

.. note:: make sure you use binary_accuracy when using binary_cross_entropy as the loss function
