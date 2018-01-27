
Loss functions in astroNN
==========================

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
            \hat{y_i}-y_i & \text{ for } y_i \neq \text{Magic Number}\\
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

.. note:: If you don't know or don't have the labels variance, you can just supply an array of zero as your labels error and let BNN deals with predictive variance

Categorical Cross-Entropy
----------------------------

Categorical Cross-Entropy is based on the equation

.. math::

   \hat{y_i} = \begin{cases}
        \begin{split}
            \epsilon & \text{ for } y_i < \epsilon \\
            1 - \epsilon & \text{ for } y_i > 1 - \epsilon \\
            \hat{y_i} & \text{ otherwise }
        \end{split}
    \end{cases}

   Loss_i = \begin{cases}
        \begin{split}
            y_i \log{\hat{y_i}} & \text{ for } y_i \neq \text{Magic Number}\\
            0 & \text{ for } y_i = \text{Magic Number}
        \end{split}
    \end{cases}

And thus the loss for mini-batch is

.. math::

   Loss_{BNN} = \frac{1}{D} \sum_{i=1}^{batch} Loss_i

Binary Cross-Entropy
----------------------------
