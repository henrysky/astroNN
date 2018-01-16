
astroNN Bayesian Convolutional Neural Net
-----------------------------------------------

Intro here

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