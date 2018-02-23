
Basic astroNN Usage
=======================================================

Workflow of setting up astroNN Neural Nets Instances and Training
--------------------------------------------------------------------

Generally, you have to setup an instances of astroNN Neural Nets class. For example,

.. code-block:: python

    # import the neural net class from astroNN first
    from astroNN.models import Apogee_CNN

    # astronn_neuralnet is an astroNN's neural network instance
    # In this case, it is an instance of APOGEE_CNN
    astronn_neuralnet = Apogee_CNN()

Lets say you have your training data prepared, you should specify what the neural network is outputing by setting up the `targetname`

.. code-block:: python

    # Just an example, if the training data is Teff, logg, Fe and absmag
    astronn_neuralnet.targetname = ['teff', 'logg', 'Fe', 'absmag']

By default, astroNN will generate folder name automatically with naming scheme `astroNN_[month][day]_run[run number]`.
But you can specify custom name by

.. code-block:: python

    # astronn_neuralnet is an astroNN's neural network instance
    astronn_neuralnet.folder_name = 'some_custom_name'

You can enable autosave (save all stuffs immediately after training or save it yourself by

.. code-block:: python

    # To enable autosave
    astronn_neuralnet.autosave = True

    # To save all the stuffs, plot=True to plot models too, otherwise wont plot, needs pydot_ng and graphviz
    astronn_neuralnet.save(plot=False)

astroNN will normalize your data after you called `train()` method. The advantage of it is if you are using normalization
provided by astroNN, you can make sure when `test()` method is called, the testing data will be normalized and prediction will
be denormalized in the exact same way as training data. This can minimize human error.

If you want to normalize by yourself, you can disable it by

.. code-block:: python

    # astronn_neuralnet is an astroNN's neural network instance
    astronn_neuralnet.input_norm_mode=0
    astronn_neuralnet.labels_norm_mode = 0

For hardware related parameters, you can force astroNN to use CPU (If you have a compatible GPU) or you can reduce/set a limit
of GPU memory usage by

.. code-block:: python

    # =================CPU related================= #
    # set cpu_fallback to fall back to CPU, no effect if you are using tensorflow instead of tensorflow-gpu
    astronn_neuralnet.cpu_fallback = True

    # =================GPU related================= #
    # set true to dynamically allocate memory
    astronn_neuralnet.limit_gpu_mem = True
    # OR enter a float smaller then 1. to set the maximum ratio of GPU memory to use
    astronn_neuralnet.limit_gpu_mem = 0.5
    # OR set None to let astroNN pre-occupy all of available GPU memory
    astronn_neuralnet.limit_gpu_mem = None

    # to debug hardware related issue
    # log_device_placement will enable tensorflow  to find out which devices your operations and tensors are assigned to
    astronn_neuralnet.log_device_placement = True

So now everything is set up for training

.. code-block:: python

    # Start the training
    astronn_neuralnet.train(x_train,y_train)

If you did not enable autosave, you can save it after training by

.. code-block:: python

    # To save all the stuffs, plot=True to plot models too, otherwise wont plot, needs pydot_ng and graphviz
    astronn_neuralnet.save(plot=False)

Load astroNN Generated Folders
-------------------------------------

First way to load a astroNN generated folder, you can use the following code. You need to replace `astroNN_0101_run001`
with the folder name. should be something like `astroNN_[month][day]_run[run number]`

.. code-block:: python

    from astroNN.models import load_folder
    astronn_neuralnet = load_folder('astroNN_0101_run001')

.. image:: openfolder_m1.png

OR second way to open astroNN generated folders is to open the folder and run command line window inside there, or switch
directory of your command line window inside the folder and run

.. code-block:: python

    from astroNN.models import load_folder
    astronn_neuralnet = load_folder()

.. image:: openfolder_m2.png

`astronn_neuralnet` will be an astroNN neural network object in this case.
It depends on the neural network type which astroNN will detect it automatically,
you can access to some methods like doing inference or continue the training (fine-tuning).
You should refer to the tutorial for each type of neural network for more detail.

There is a few parameters from keras_model you can always access,

.. code-block:: python

    # The model summary from Keras
    astronn_neuralnet.keras_model.summary()

    # The model input
    astronn_neuralnet.keras_model.input

    # The model input shape expectation
    astronn_neuralnet.keras_model.input_shape

    # The model output
    astronn_neuralnet.keras_model.output

    # The model output shape expectation
    astronn_neuralnet.keras_model.output_shape


astroNN neuralnet object also carries `targetname` (hopefully correctly set by the writer of neural net), parameters
used to normalize the training data (The normalization of training and testing data must be the same)

.. code-block:: python

    # The tragetname corresponding to output neurone
    astronn_neuralnet.targetname

    # The model input
    astronn_neuralnet.keras_model.input

    # The mean used to normalized training data
    astronn_neuralnet.input_mean_norm

    # The standard derivation used to normalized training data
    astronn_neuralnet.input_std_norm

    # The mean used to normalized training labels
    astronn_neuralnet.labels_mean_norm

    # The standard derivation used to normalized training labels
    astronn_neuralnet.labels_std_norm

Workflow of testing and distributing astroNN models
-------------------------------------------------------

The first step of the workflow should be loading an astroNN folder as described above.

Lets say you have loaded the folder and have some testing data, you just need to provide the testing data without
any normalization if you used astroNN normalization during training. The testing data will be normalized and prediction will
be denormalized in the exact same way as training data.

.. code-block:: python

    # Run forward pass for the test data throught the neural net to get prediction
    # The prediction should be denormalized if you use astroNN normalization during training
    prediction = astronn_neuralnet.test(x_test)

You can always train on new data based on existing weights (NOT recommended as I am still trying to fix some issues)

.. code-block:: python

    # Start the training on existing models (fine-tuning)
    astronn_neuralnet.train(x_train,y_train)