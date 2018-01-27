
Loss functions
==================

astroNN provides modified loss function built on Tensorflow, which is similar to Keras. All astroNN loss functions are
fully compatible with Keras with Tensorflow backend.

Regression-related
-------------------------------------

mean_squared_error
++++++++++++++++++++++

Mean Squared Error is based on the equation

.. math::

   \[
   \text{loss} = \begin{cases}
        \begin{split}
            (\hat{y}-y)^2, \text{for y} \neq \text{Magic Number}\\
            0, \text{for y} = \text{Magic Number}
        \end{split}
    \end{cases}
\]


.. code:: python

    from astroNN.models import load_folder
    astronn_neuralnet = load_folder('astroNN_0101_run001')

astronn_neuralnet will be an astroNN neural network object in this case.
It depends on the neural network type which astroNN will detect it automatically,
you can access to some methods like doing inference or continue the training (fine-tuning).
You should refer to the tutorial for each type of neural network for more detail.

.. code:: python

    astronn_neuralnet.test(x_test, y_test)

Classification-related
--------------------------------------

All astroNN Neural Nets are inherited from some child classes which inherited NeuralNetMaster

::

    NeuralNetMaster
    ├── CNNBase
    │   ├── Apogee_CNN
    │   ├── StarNet2017
    │   └── Cifar10
    ├── BayesianCNNBase
    │   └── Apogee_BCNN
    ├── ConvVAEBase
    │   └── APGOEE_CVAE
    └── CGANBase