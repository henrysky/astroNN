
Basic Usage and Introduction to astroNN Neural Nets
=======================================================

Load astroNN Generated Folders
-------------------------------------

First way to load a astroNN generated folder, you can use the following code. You need to replace 'astroNN_0101_run001'
with the folder name. should be something like 'astroNN_[month][day]_run[run number]'

.. code-block:: python

    from astroNN.models import load_folder
    astronn_neuralnet = load_folder('astroNN_0101_run001')

Second way to open astroNN generated folders is to open the folder and run command line window there, or switch
directory of your command line window inside the folder and run

.. code-block:: python

    from astroNN.models import load_folder
    astronn_neuralnet = load_folder()

astronn_neuralnet will be an astroNN neural network object in this case.
It depends on the neural network type which astroNN will detect it automatically,
you can access to some methods like doing inference or continue the training (fine-tuning).
You should refer to the tutorial for each type of neural network for more detail.

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
