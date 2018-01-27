
Loss functions
==================

astroNN provides modified loss function which is capable to deal with missing labels (represented by Magic Number).
Since they are similar to Keras and built on Tensorflow, all astroNN loss functions are fully compatible with Keras with
Tensorflow backend.

Mean Squared Error
-----------------------

MSE is  is based on the equation

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

Regression Loss and Predictive Variance Loss for Bayesian Neural Net
------------------------------------------------------------------------

It is based on the equation, please notice :math:`s` is :math:`log((\sigma_predictive)^2 + (\sigma_{known})^2)`
to avoid numerical instability

.. math::

   Loss_i = \begin{cases}
        \begin{split}
            \frac{1}{2} (\hat{y_i}-y_i)^2 e^{-\text{s_i}} + \frac{1}{2}(s_i) & \text{ for } y_i \neq \text{Magic Number}\\
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
