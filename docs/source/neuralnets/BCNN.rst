.. astroNN documentation master file, created by
   sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bayesian Convolutional Neural Network class
--------------------------------------------

astroNN.models.BCNN is a 4 layered convolutional neural net (2 convolutional layers and 2 dense layers) with dropout and l2 regularizers in every layers. 

You can create Bayesian CNN in astroNN using

.. code:: python

    from astroNN.models import BCNN

    # And then create an object of Convolutional Neural Network classs
    bcnn_net = BCNN()