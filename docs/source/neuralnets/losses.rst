
Loss functions
==================

astroNN provides modified loss function built on Tensorflow, which is similar to Keras. All astroNN loss functions are
fully compatible with Keras with Tensorflow backend.

Mean Squared Error
-----------------------

Mean Squared Error can be imported by

.. code:: python

    from astroNN.models.losses import mean_squared_error

which is  is based on the equation

.. math::

   \text{loss_i} = \begin{cases}
        \begin{split}
            (\hat{y_i}-y_i)^2 & \text{ for y_i} \neq \text{Magic Number}\\
            0 & \text{ for y_i} = \text{Magic Number}
        \end{split}
    \end{cases}

   \text{L\textsubscript{NN}} = \frac{1}{D} \sum_{i=1}^{batch size} \text{loss_i}

Regression Loss for Bayesian Neural Net
-------------------------------------------

Regression Loss for Bayesian Neural Net can be imported by

.. code:: python

    from astroNN.models.losses import mse_lin_wrapper

which is based on the equation, please notice :math:`s` is :math:`log(\sigma^2)` to avoid numerical instability

.. math::

   \text{loss} = \begin{cases}
        \begin{split}
            \frac{1}{2} (\hat{y}-y)^2 e^{-\text{s}} + \frac{1}{2}(\text{s}) & \text{ for y} \neq \text{Magic Number}\\
            0 & \text{ for y} = \text{Magic Number}
        \end{split}
    \end{cases}

Regression Loss for predictive variance for Bayesian Neural Net
------------------------------------------------------------------

Regression Loss for predictive variance for Bayesian Neural Net can be imported

.. code:: python

    from astroNN.models.losses import mse_var_wrapper

which is based on the equation, please notice :math:`s` is :math:`log(\sigma^2)` to avoid numerical instability

.. math::

   \text{loss} = \begin{cases}
        \begin{split}
            \frac{1}{2} (\hat{y}-y)^2 e^{-\text{s}} + \frac{1}{2}(\text{s}) & \text{ for y} \neq \text{Magic Number}\\
            0 & \text{ for y} = \text{Magic Number}
        \end{split}
    \end{cases}
