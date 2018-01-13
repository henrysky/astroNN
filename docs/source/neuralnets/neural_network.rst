.. astroNN documentation master file, created by
sphinx-quickstart on Thu Dec 21 17:52:45 2017.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

astroNN Neural Network class
=============================

Loading a astroNN generated folder
-----------------------------------

To load a astroNN generated folder, you can use the following code. You need to replace 'astroNN_0101_run001' with the folder name. should be something like 'astroNN_[month][day]_run[run number]'

.. code:: python

    from astroNN.models import load_folder
    astronn_neuralnet = load_folder('astroNN_0101_run001')

astronn_neuralnet will be an astroNN neural network object in this case. It depends on the neural network type which astroNN will detect it automatically, you can access to some methods like doing inference or continue the training (fine-tuning). You should refer to the tutorial for each type of neural network for more detail.

.. code:: python

    astronn_neuralnet.test(x_test, y_test)