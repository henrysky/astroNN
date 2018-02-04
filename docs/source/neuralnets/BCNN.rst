
Bayesian Convolutional Neural Net
======================================

.. warning:: This is a draft


With traditional neural network, weight in neural network are point estimate which result a point estimate result.
Unlike statistical modelling which have uncertainty estimates, the whole point of machine learning is just learn from
data and predict an single outcome. Uncertainty estimates is important in astronomy and it will be best if we could
add uncertainty to neural network.

Bayes Rule
-------------

To understand how a Bayesian Neural Net works, we must first known about Bayesian statistics. The core of Bayesian
statistic is Bayes Rule.

Suppose we have event A and B. Bayes Rule tells us :math:`P(A|B)=\frac{P(B|A)P(A)}{P(B)}` where :math:`P(A|B)` is
conditional probability which represents the likelihood of event A occurring given that B occurred. :math:`P(B|A)`
represents the likelihood of event B occurring given that A occurred. :math:`P(A)` and :math`P(B)` are probability of
observing A and B independently of each other.

The Bayesian interpretation of a probablility is a measure of a prior belief. In such case, :math:`P(A)` can be viewed
as a prior belief in A and :math:`P(A|B)` measures the postterior belief of having accounted for B.

Simple Bayesian Regression
-------------------------------

The problem is a linear regression problem, we have some input data :math:`X` and output data :math:`Y` and we
want to find :math:`w` such that :math:`Y = wX`. Suppose we use Mean Squared Error (L2) loss which is commonly found in
neural network. The objective :math:`(Y-wX)^2`

First step, we need to somehow change this to a probability. You want to maximizing the
likelihood to generate :math:`Y` given you have :math:`X` and :math:`w`, i.e. :math:`P(Y|X,w)`

Please notice using Mean Squared Error (L2), it is equivalent maximizing the log-likelihood of a Gaussian, i.e :math:`Y`
is Gaussian distributed.

But we want this problem be Bayesian, so we impose a prior belief on our weight, :math:`P(Y|X,w) P(w)`.
Usually we set gaussian distribution as our belief.

By Bayes Rule, the posterior distribution of the weight is :math:`P(w|X,Y)=\frac{P(Y|X,w)P(w)}{C}` and
:math:`C` is :math:`P(Y)` or :math:`\int P(X, w) dw`, an integral usually very difficult to calculate.

Variational Inference
--------------------------

To solve this problem we will need to use Variational Inference. How to do Variational Inference.

The first step we need to introduce a parameterised distribution :math:`Q(w|v)`, Q representing a variational
distribution and :math:`v` is the variational parameter, over :math:`w` to approximate the true posterior.

And bingo, another advantage is from an integration problem, we now have an optimizing problem on variational parameter
:math:`v`. What are we optimizing to? We need to have a :math:`v` so that to match the true posterior distribution as
good as possible. True posterior refers to :math:`P(w|y,x)` and of course we better have a :math:`Q(w|v)` which close
to the true posterior.

Approximation to the integral of a probability distribution (:math:`\int P(X, w) dw` in this case) can be done by Monte
Carlo Sampling (similarilty to estimation of :math:`\pi` by MC sampling)

Full Bayesian way of doing Bayesian Neural Net
--------------------------------------------------

First we need to place a prior on the weight by getting every weight from gaussian distribution center at 0 with scale 1.

astroNN Bayesian Neural Network by Dropout Variational Inference
-------------------------------------------------------------------

The core idea astroNN Bayesian Neural Network is Neural Net with Dropout Variational Inference and gaussian prior
weights is a bayesian approximation of gaussian process.

Still in progress

How does astroNN calculate uncertainty from neural network for regression task
--------------------------------------------------------------------------------

.. math::

   \text{Prediction} = \text{Mean from Dropout Variational Inference}

.. math::

   \text{Total Variance} = \text{Variance from Dropout Variational Inference} + \text{Mean of Predictive Variance Output} + \text{Inverse Model Precision}

.. math::

   \text{Prediction with Error} = \text{Prediction} \pm \sqrt{\text{Total Variance}}

Inverse Model Precision is by definition

.. math::

   \tau ^{-1} = \frac{2N \lambda}{l^2 p}
    \text{where } \lambda \text{ is the l2 regularization parameter, l is scale length, p is the probability of a neurone NOT being dropped and N is total training data}

For more detail, please see my demonstration here_

.. _here: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis