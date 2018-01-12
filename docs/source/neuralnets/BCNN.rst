.. astroNN documentation master file, created by
   sphinx-quickstart on Thu Dec 21 17:52:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

astroNN Bayesian Convolutional Neural Network class
----------------------------------------------------

astroNN.models.BayesianCNNBase is an abstract class, you can inherit it to create a Bayesian Convolutional Neural Network easily

You can create Bayesian CNN in astroNN using

.. code:: python

    from astroNN.models import BCNN

    # And then create an object of Convolutional Neural Network classs
    bcnn_net = BCNN()