
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

Suppose we have event A and B. Bayes Rule tells us :math:`P(A|B)=\frac{P(B|A)P(A)}{P(B)}` where :math:`P(A|B)` is
conditional probability which represents the likelihood of event A occuring given that B occurred. :math:`P(B|A)`
represents the likelihood of event B occuring given that A occurred. :math:`P(A)` and :math`P(B)` are probability of
observing A and B independently of each other.

The Bayesian interpretation of a probablility is a measure of a prior belief. In such case, :math:`P(A)` can be viewed
as a prior belief in A and :math:`P(A|B)` measures the postterior belief of having accounted for B.

Simple Bayesian Regression
=============================

The problem is a linear regression problem, we have some input data :math:`X` and output data :math:`Y` and we
want to find :math:`w` such that :math:`Y = wX`. Suppse we use Mean Squared Error loss which is commonly found in
neural network. The objective :math:`(Y-wX)^2`

First step, we need to somehow change this to a probability. You want to maximizing the
likelihood to generate :math:`Y` given you have :math:`X` and :math:`w`, i.e. :math:`P(Y|X,w)`

Please notice using Mean Squared Error, it is equivalent maximizing the log-likelihood of a Gaussian, i.e :math:`Y` is
Gaussian distributed.

But we want this problem be Bayesian, so we impose a prior belief on our weight, :math:`P(Y|X,w) P(w)`
Usually we set gaussian distribution as our belief.

By Bayes Rule, the posterior distribution of the weight is :math:`P(w|X,Y)=\frac{P(Y|X,w)P(w)}{C}` and
:math:`C` is :math:`\int P(X, w) dw`, an integral usually very difficult to calculate.

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