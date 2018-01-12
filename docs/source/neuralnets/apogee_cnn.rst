
APOGEE Spectra analysis using Convolutional Neural Net
---------------------------------------------------------

Althought in theory you can feed any 1D data to astroNN neural networks. This tutorial will only focus on spectra analysis.

.. code:: python

    from astroNN.models import APOGEE_CNN
    from astroNN.datasets import H5Loader

    # Load the train data from dataset first, x_train is spectra and y_train will be ASPCAP labels
    loader = H5Loader('datasets.h5')
    loader.load_err = False
    x_train, y_train = loader.load()

    # And then create an object of Convolutional Neural Network classs
    cnn_net = APOGEE_CNN()

    # You dont have to specify the task because its 'regression' by default. But if you are doing classification. you can set task='classification'
    cnn_net.task = 'regression'

    # Set max_epochs to 10 for a quick result. You should train more epochs normally
    cnn_net.max_epochs = 10
    cnn_net.train(x_train, y_train)


Here is a list of parameter you can set but you can also not set them to use default

.. code:: python

    CNN.batch_size = 64
    CNN.initializer = 'he_normal'
    CNN.activation = 'relu'
    CNN.num_filters = [2, 4]
    CNN.filter_length = 8
    CNN.pool_length = 4
    CNN.num_hidden = [196, 96]
    CNN.max_epochs = 250
    CNN.lr = 0.005
    CNN.reduce_lr_epsilon = 0.00005
    CNN.reduce_lr_min = 0.0000000001
    CNN.reduce_lr_patience = 10
    CNN.fallback_cpu = False
    CNN.limit_gpu_mem = True
    CNN.target = 'all'
    CNN.l2 = 1e-7
    CNN.dropout_rate = 0.2
    CNN.length_scale = 1.0  # prior length scale
    CNN.input_norm_mode = 1
    CNN.labels_norm_mode = 2

.. note:: You can disable astroNN data normalization via CNN.input_norm_mode=0 as well as CNN.labels_norm_mode = 2 and do normalization yourself. But make sure you dont normalize labels with -9999 (missing labels).

After the training, you can use 'cnn_net' in this case and call test method to test the neural network on test data. Or you can load the folder by

.. code:: python

    from astroNN.models import load_folder
    cnn_net = load_folder('astroNN_0101_run001')

    # Load the test data from dataset, x_test is spectra and y_test will be ASPCAP labels
    loader2 = H5Loader('datasets.h5')
    loader2.load_combined = False
    x_test, y_test = loader2.load()

    pred = cnn_net.test(x_test)  # pred contains denormalized result aka. ASPCAP labels prediction in this case


Since astroNN.models.CNN does not have uncertainty analysis feature. You can plot aspcap label residue by supplying zeros arrays as error value. If you want model uncertainty/ risk estimation and propagated error, please use astroNN.models.BCNN.

.. code:: python

   import numpy as np
   cnn_net.aspcap_residue_plot(pred, y_test, np.zeros(y_test.shape))


You can calculate jacobian which represents the output derivative to the input and see where those output is sensitive to in inputs.

.. code:: python

   cnn_net.jacobian(x_test)

.. note:: You can access to Keras model method like model.predict via (in the above tutorial) cnn_net.keras_model (Example: cnn_net.keras_model.predict())

Example Plots using aspcap_residue_plot
============================================

.. image:: /neuralnets/cnn_apogee/logg_test.png
.. image:: /neuralnets/cnn_apogee/teff_test.png

ASPCAP labels prediction using CNN vs The Cannon 2
===================================================

.. image:: https://image.ibb.co/fDY5JG/table1.png

Example Plots using jacobian
============================================

.. image:: /neuralnets/bcnn_apogee/Cl_jacobian.png
.. image:: /neuralnets/bcnn_apogee/Na_jacobian.png