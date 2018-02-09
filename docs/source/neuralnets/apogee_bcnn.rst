APOGEE Spectra analysis using Bayesian Convolutional Neural Net
-----------------------------------------------------------------

Althought in theory you can feed any 1D data to astroNN neural networks. This tutorial will only focus on spectra analysis.

.. code-block:: python

    from astroNN.models import Apogee_BCNN
    from astroNN.datasets import H5Loader

    # Load the train data from dataset first, x_train is spectra and y_train will be ASPCAP labels
    loader = H5Loader('datasets.h5')
    loader2.load_combined = True
    loader2.load_err = True
    x_train, y_train, x_err, y_err = loader.load()

    # And then create an object of Bayesian Convolutional Neural Network classs
    bcnn_net = Apogee_BCNN()

    # You dont have to specify the task because its 'regression' by default. But if you are doing classification. you can set task='classification'
    bcnn_net.task = 'regression'

    # Set max_epochs to 10 for a quick result. You should train more epochs normally, especially with dropout
    bcnn_net.max_epochs = 10
    bcnn_net.train(x_train, y_train, x_err, y_err)

Here is a list of parameter you can set but you can also not set them to use default

.. code-block:: python

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
    BCNN.target = 'all'
    BCNN.l2 = 1e-7
    BCNN.dropout_rate = 0.2
    BCNN.length_scale = 0.1  # prior length scale
    BCNN.input_norm_mode = 1
    BCNN.labels_norm_mode = 2

.. note:: You can disable astroNN data normalization via BCNN.input_norm_mode=0 as well as BCNN.labels_norm_mode = 0 and do normalization yourself. But make sure you dont normalize labels with MAGIC_NUMBER (missing labels).

After the training, you can use 'bcnn_net' in this case and call test method to test the neural network on test data. Or you can load the folder by

.. code-block:: python

    from astroNN.models import load_folder
    bcnn_net = load_folder('astroNN_0101_run001')

    # Load the test data from dataset, x_test is spectra and y_test will be ASPCAP labels
    loader2 = H5Loader('datasets.h5')
    loader2.load_combined = False
    loader2.load_err = True
    x_test, y_test, x_err, y_err = loader2.load()

    # pred contains denormalized result aka. ASPCAP labels prediction in this case
    # pred_std is the total uncertainty (standard derivation) which is the sum of all the uncertainty
    # predictive_std is the predictive uncertainty predicted by bayesian neural net
    # model_std is the model uncertainty from dropout variational inference
    pred, pred_std, predictive_std, model_std = bcnn_net.test(x_test, x_err)


Since astroNN.models.BCNN uses Bayesian deep learning which provides uncertainty analysis features. If you want quick testing/prototyping, please use astroNN.models.CNN. You can plot aspcap label residue by

.. code-block:: python

   bcnn_net.aspcap_residue_plot(pred, y_test, pred_std)


You can calculate jacobian which represents the output derivative to the input and see where those output is sensitive to in inputs.

.. code-block:: python

    # Calculate jacobian first
    jacobian_array = bcnn_net.jacobian(x_test, mean_output=True)

    # Plot the graphs
    bcnn_net.jacobian_aspcap(jacobian=jacobian_array, dr=14)

.. note:: You can access to Keras model method like model.predict via (in the above tutorial) bcnn_net.keras_model (Example: bcnn_net.keras_model.predict())

ASPCAP Labels Prediction
===========================

Training set (approx. 32000 spectra): Starflag and aspcap flag cuts, 4000<Teff<5500, SNR<200, must be combined spectra

Testing set (approx. 90000 spectra): Individual Visit of the training spectra

Ground Truth is ASPCAP labels.

+-------------+---------------------+-------------------------------+
|             | Median of residue   | astropy mad_std of residue    |
+=============+=====================+===============================+
| Al          | -0.004              | 0.044                         |
+-------------+---------------------+-------------------------------+
| Alpha       |  0.000              | 0.014                         |
+-------------+---------------------+-------------------------------+
| C           |  0.000              | 0.034                         |
+-------------+---------------------+-------------------------------+
| C1          |  0.004              | 0.039                         |
+-------------+---------------------+-------------------------------+
| Ca          | -0.003              | 0.023                         |
+-------------+---------------------+-------------------------------+
| Co          | -0.009              | 0.071                         |
+-------------+---------------------+-------------------------------+
| Cr          | -0.002              | 0.031                         |
+-------------+---------------------+-------------------------------+
| fakemag     |  0.324              | 6.112                         |
+-------------+---------------------+-------------------------------+
| Fe          | -0.001              | 0.018                         |
+-------------+---------------------+-------------------------------+
| K           | -0.002              | 0.033                         |
+-------------+---------------------+-------------------------------+
| Log(g)      |  0.001              | 0.058                         |
+-------------+---------------------+-------------------------------+
| M           |  0.000              | 0.016                         |
+-------------+---------------------+-------------------------------+
| Mg          |  0.001              | 0.022                         |
+-------------+---------------------+-------------------------------+
| Mn          | -0.002              | 0.027                         |
+-------------+---------------------+-------------------------------+
| N           |  0.003              | 0.040                         |
+-------------+---------------------+-------------------------------+
| Na          | -0.001              | 0.106                         |
+-------------+---------------------+-------------------------------+
| Ni          |  0.000              | 0.023                         |
+-------------+---------------------+-------------------------------+
| O           | -0.001              | 0.029                         |
+-------------+---------------------+-------------------------------+
| P           |  0.004              | 0.090                         |
+-------------+---------------------+-------------------------------+
| S           |  0.012              | 0.043                         |
+-------------+---------------------+-------------------------------+
| Si          |  0.001              | 0.024                         |
+-------------+---------------------+-------------------------------+
| Teff        | -10.602             | 47.895                        |
+-------------+---------------------+-------------------------------+
| Ti          | -0.0002             | 0.033                         |
+-------------+---------------------+-------------------------------+
| Ti2         | -0.017              | 0.091                         |
+-------------+---------------------+-------------------------------+
| V           |  0.001              | 0.060                         |
+-------------+---------------------+-------------------------------+

Example Plots using aspcap_residue_plot
============================================

.. image:: /neuralnets/bcnn_apogee/logg_test.png
.. image:: /neuralnets/bcnn_apogee/Fe_test.png

Example Plots using jacobian
============================================

.. image:: /neuralnets/bcnn_apogee/Cl_jacobian.png
.. image:: /neuralnets/bcnn_apogee/Na_jacobian.png