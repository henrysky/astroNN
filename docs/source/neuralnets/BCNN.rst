.. astroNN documentation master file, created by
   sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bayesian Convolutional Neural Network
---------------------------------------

astroNN.models.BCNN is a 4 layered convolutional neural net (2 convolutional layers and 2 dense layers) with dropout and l2 regularizers in every layers. 

You can create Bayesian CNN in astroNN using

.. code:: python

    from astroNN.models import BCNN

    # And then create an object of Convolutional Neural Network classs
    bcnn_net = BCNN()

APOGEE Spectra Analysis
--------------------------------------------------

Althought in theory you can feed any 1D data to astroNN neural networks. This tutorial will only focus on spectra analysis.

.. code:: python

    from astroNN.models import BCNN
    from astroNN.datasets import H5Loader

    # Load the train data from dataset first, x_train is spectra and y_train will be ASPCAP labels
    loader = H5Loader('datasets.h5')
    x_train, y_train = loader.load()

    # And then create an object of Bayesian Convolutional Neural Network classs
    bcnn_net = BNN()

    # You dont have to specify the task because its 'regression' by default. But if you are doing classification. you can set task='classification'
    bcnn_net.task = 'regression'

    # Set max_epochs to 10 for a quick result. You should train more epochs normally, especially with dropout
    bcnn_net.max_epochs = 10
    bcnn_net.train(x_train, y_train)

Here is a list of parameter you can set but you can also not set them to use default

.. code:: python

    BCNN.batch_size = 64
    BCNN.initializer = 'he_normal'
    BCNN.activation = 'relu'
    BCNN.num_filters = [2, 4]
    BCNN.filter_length = 8
    BCNN.pool_length = 4
    BCNN.num_hidden = [196, 96]
    BCNN.max_epochs = 250
    BCNN.lr = 0.005
    BCNN.reduce_lr_epsilon = 0.00005
    BCNN.reduce_lr_min = 0.0000000001
    BCNN.reduce_lr_patience = 10
    BCNN.fallback_cpu = False
    BCNN.limit_gpu_mem = True
    BCNN.data_normalization = True
    BCNN.target = 'all'
    BCNN.l2 = 1e-7
    BCNN.dropout_rate = 0.2
    BCNN.length_scale = 1.0  # prior length scale

.. note:: You can disable astroNN data normalization via bcnn_net.data_normalization=False and do normalization yourself. But make sure you dont normalize labels with -9999 (missing labels).

After the training, you can use 'bcnn_net' in this case and call test method to test the neural network on test data. Or you can load the folder by

.. code:: python

    from astroNN.models import load_folder
    bcnn_net = load_folder('astroNN_0101_run001')

    # Load the test data from dataset, x_test is spectra and y_test will be ASPCAP labels
    loader2 = H5Loader('datasets.h5')
    loader2.load_combined = False
    x_test, y_test = loader2.load()

    pred, pred_var = bcnn_net.test(x_test)  # pred contains denormalized result aka. ASPCAP labels prediction in this case


Since astroNN.models.BCNN uses Bayesian deep learning which provides uncertainty analysis features. If you want quick testing/prototyping, please use astroNN.models.CNN. You can plot aspcap label residue by

.. code:: python

   bcnn_net.aspcap_residue_plot(pred, y_test, pred_var)


You can calculate jacobian which represents the output derivative to the input and see where those output is sensitive to in inputs.

.. code:: python

   bcnn_net.jacobian(x_test)

.. note:: You can access to Keras model method like model.predict via (in the above tutorial) bcnn_net.keras_model (Example: bcnn_net.keras_model.predict())

How does astroNN calculate uncertainty from neural network
============================================================

.. math::

   \text{Prediction} = \text{Mean from Variational Inference by Dropout}

.. math::

   \text{Total Variance} = \text{Variance from Variational Inference by Dropout} + \text{Predictive Variance Output} + \text{Inverse Model Precision}

.. math::

   \text{Prediction with Error} = \text{Prediction} \pm \sqrt{\text{Total Variance}}

Inverse Model Precision is by definition

.. math::

   \tau ^{-1} = \frac{2N \lambda}{l^2 p}, \text{where } \lambda \text{ is the l2 regularization parameter, l is scale length, p is the probability of a neurone NOT being dropped and N is total training data}

For more detail, please see my demonstration here_

.. _here: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis

Example Plots using aspcap_residue_plot
============================================

.. image:: /neuralnets/bcnn_apogee/logg_test.png
.. image:: /neuralnets/bcnn_apogee/teff_test.png

Example Plots using jacobian
============================================

.. image:: /neuralnets/bcnn_apogee/Cl_jacobian.png
.. image:: /neuralnets/bcnn_apogee/Na_jacobian.png