.. automodule:: astroNN.neuralode

NeuralODR - **astroNN.neuralODE**
===================================

Neural ODE (Neural Ordinary Differential Equation) module provides numerical integrator implemented in ``Tensorflow``
for solutions of an ODE system, and can calculate gradient.

Numerical Integrator
-----------------------

``astroNN`` implemented numerical integrator in ``Tensorflow``

.. automodule:: astroNN.neuralode.odeint

.. autofunction::  astroNN.neuralode.odeint.odeint

An example integration an ODE for ``sin(x)``

.. code-block:: python

    import time
    import numpy as np
    import tensorflow as tf
    from astroNN.shared.nn_tools import cpu_fallback, gpu_memory_manage
    from astroNN.neuralode import odeint

    cpu_fallback()
    gpu_memory_manage()

    # time array
    t = tf.constant(np.linspace(0, 100, 10000))
    # initial condition
    true_y0 = tf.constant([0., 5.])
    # analytical ODE system for sine wave [x, t] -> [v, a]
    ode_func = lambda y, t: tf.stack([tf.cos(5*t), tf.sin(5*t)])

    start_t = time.time()
    true_y = odeint(ode_func, true_y0, t, method='dop853')
    print(time.time() - start_t)  # approx. 4.3 seconds on i7-9750H GTX1650

Moreover ``odeint`` supports numerically integration in parallel, the example below integration the ``sin(x)`` for 50 initial
conditions. You can see the execution time is the same!!

.. code-block:: python

    start_t = time.time()
    # initial conditions, 50 of them instead of a single initial condition
    true_y0sss = tf.random.normal((50, 2), 0, 1)
    true_y = odeint(ode_func, true_y0sss, t, method='dop853')
    print(time.time() - start_t)  # also approx. 4.3 seconds on i7-9750H GTX1650
