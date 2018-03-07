
Bayesian Neural Net with Dropout Variational Inference
============================================================

.. warning:: This is a draft


With traditional neural network, weight in neural network are point estimate which result a point estimate result.
Unlike statistical modelling which have uncertainty estimates, the whole point of machine learning is just learn from
data and predict an single outcome. Uncertainty estimates is important in astronomy and it will be best if we could
add uncertainty to neural network.

Background Knowledge
-----------------------

To understand Bayesian Neural Net, we first need to understand some background knowledge.

-------------
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

-------------------------------
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

--------------------------
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

Dropout Variational Inference
--------------------------------

The core idea Bayesian Neural Network is Neural Net with Dropout Variational Inference and gaussian prior
weights is bayesian. By reparametrising the approximate variational distribution `Q(w|v)` to be Bernoulli

.. math::

   r_{i} = \text{Bernoulli} (p) \\
   \hat{y_i} = r_{i} * y_i

which is exactly the thing used by dropout.

Thus the loss is

.. math::

   \mathcal{L}_{dropout} = \frac{1}{D} \sum_{i=1}^{batch} (Loss_i) + \lambda \sum_{i=1}^{Layer} (Weight)^2


How is uncertainty calculated from neural network for regression task
--------------------------------------------------------------------------------

.. math::

   \text{Prediction} = \text{Mean from Dropout Variational Inference}

.. math::

   \text{Total Variance} = \text{Variance from Dropout Variational Inference} + \text{Mean of Predictive Variance Output} + \text{Inverse Model Precision}

Or if you have known input data uncertainty, you should add the propagated uncertainty to the final variance too.

The final prediction will be

.. math::

   \text{Prediction with Error} = \text{Prediction} \pm \sqrt{\text{Total Variance}}

Inverse Model Precision is by definition

.. math::

   \tau ^{-1} = \frac{2N \lambda}{l^2 p}
    \text{where } \lambda \text{ is the l2 regularization parameter, l is scale length, p is the probability of a neurone NOT being dropped and N is total training data}

For more detail, please see my demonstration here_

.. _here: https://github.com/henrysky/astroNN/tree/master/demo_tutorial/NN_uncertainty_analysis


A simple way to think about predictive, model and propagated uncertainty
--------------------------------------------------------------------------

Since Bayesian Neural Network involves multiple source of uncertainty and they can be confusing. There is one simple way
to think about these uncertainty.

Let's say you have a student and some maths problems with solutions and some maths problems without solutions. For simplicity
all the maths problems are only either differentiation or integration. You want the solution for those maths problems without
solution. One way to do it is to let the student to do the maths with known solution, and evaluate his/her performance.
If the student did all the integration problems wrong, then you know the integration solutions from the student cannot be trusted.

In more real life situation, you don't know the training process/data, but you can interact with a trained student. Now you
just give an integration problem to the student, the student should tells you he/she does not have confidence on that
problem at all because it is about integration and the student knows his/her own ability for doing integration poorly.
This is something that is predictable, so we call them predictive uncertainty.

Let's say the student has done very well on differentiation problems and you should expect he/she has a high confidence
on this area. But if you are a teacher, you know if students said they understand a topic, they probably not really understand it.
One way to measure the model uncertainty from the student is you give the problems to the student to solve and you get back a set of
solutions. And after a week or so, you give the same problems to the student to solve and you get another set of solutions. If the
two solutions are the same, and the student said he/she is confident, then you know the solutions are probably right. If the
two solutions are not the same, then even the student said he/she is confident, you should not trust those solutions from
the student.

The propagated uncertainty can be just as simple as you have some typos in the problems, and lead to the student giving some
wrong answers.