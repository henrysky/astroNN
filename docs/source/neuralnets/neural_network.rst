
Basic Usage and Introduction to astroNN Neural Nets
=======================================================

Load astroNN Generated Folders
-------------------------------------

First way to load a astroNN generated folder, you can use the following code. You need to replace 'astroNN_0101_run001'
with the folder name. should be something like 'astroNN_[month][day]_run[run number]'

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

astronn_neuralnet will be an astroNN neural network object in this case.
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

Available astroNN Neural Net Classes
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
        └── GalaxyGAN2017

NeuralNetMaster Class
--------------------------------------

NeuralNetMaster is the top level abstract class for all astroNN sub neural network class. NeuralNetMaster define the
structure of how an astroNN neural network class should look like.

NeuralNetMaster consists of a pre-training checking (check input and labels shape, cpu/gpu check and create astroNN
folder for every run

CNNBase Class
--------------------------------------
