
Bayesian Convolutional Neural Net
-------------------------------------

With traditional neural network, weight in neural network are point estimate which result a point estimate result.
Unlike statistical modelling which have uncertainty estimates, the whole point of machine learning is just learn from
data and predict an single outcome. Uncertainty estimates is important in astronomy and it will be best if we could
add uncertainty to neural network.

Bayes Rule
===========

To understand how a Bayesian Neural Net works, we must first known about Bayesian statistics. The core of Bayesian
statistic is Bayes Rule.

Suppose we have event A and B. Bayes Rule tells us :math:`P(A|B)=\frac{P(B|A)P(A)}{P(B)}`

How does astroNN calculate uncertainty from neural network
============================================================

.. math::

   \text{Prediction} = \text{Mean from Dropout Variational Inference}

.. math::

   \text{Total Variance} = \text{Variance from Dropout Variational Inference} + \text{Mean of Predictive Variance Output} + \text{Inverse Model Precision}

.. math::

   \text{Prediction with Error} = \text{Prediction} \pm \sqrt{\text{Total Variance}}

Inverse Model Precision is by definition

.. math::

   \tau ^{-1} = \frac{2N \lambda}{l^2 p}, \text{where } \lambda \text{ is the l2 regularization parameter, l is scale length, p is the probability of a neurone NOT being dropped and N is total training data}

For more detail, please see my demonstration here_

.. _here: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis