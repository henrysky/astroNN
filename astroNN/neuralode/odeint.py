import tensorflow as tf
from astroNN.neuralode.dop853 import dop853
from astroNN.neuralode.runge_kutta import rk4


def odeint(func=None, x=None, t=None, method='dop853', precision=tf.float32, *args, **kwargs):
    """
    To computes the numerical solution of a system of first order ordinary differential equations y'=f(x,y). Default
    precision at float32.

    :param func: function of the differential equation, usually take func([position, velocity], time) and return velocity, acceleration
    :type func: callable
    :param x: initial x, usually is [position, velocity]
    :type x: Union([tf.Tensor, numpy.ndarray, list])
    :param t: set of times at which one wants the result
    :type t: Union([tf.Tensor, numpy.ndarray, list])
    :param method: numerical integrator to use, available integrators are ['dop853', 'rk4']
    :type method: str
    :param precision: float precision, tf.float32 or tf.float64
    :type precision: type
    :param t: set of times at which one wants the result
    :type t: Union([tf.Tensor, numpy.ndarray, list])

    :return: integrated result
    :rtype: tf.Tensor

    :History: 2020-May-31 - Written - Henry Leung (University of Toronto)
    """
    if method.lower() == 'dop853':
        ode_method = dop853
    elif method.lower() == 'rk4':
        ode_method = rk4
    else:
        raise NotImplementedError(f"Method {method} is not implemented")

    # check things if they are tensors
    if not isinstance(x, tf.Tensor):
        x = tf.constant(x)
    if not isinstance(t, tf.Tensor):
        t = tf.constant(t)

    if precision == tf.float32:
        tf_float = tf.float32
    elif precision == tf.float64:
        tf_float = tf.float64
    else:
        raise TypeError(f"Data type {precision} not understood")

    x = tf.cast(x, tf_float)
    t = tf.cast(t, tf_float)

    is_only_one_flag = False

    if len(x.shape) < 2:  # ensure multi-dim
        is_only_one_flag = True
        x = tf.expand_dims(x, axis=0)
        total_num = 1
    else:
        total_num = x.shape[0]

    if len(t.shape) < 2:
        t = tf.stack([t] * total_num)

    def odeint_external(tensor):
        return ode_method(func=func, x=tensor[0], t=tensor[1], precision=precision, *args, **kwargs)

    @tf.function
    def parallelized_func(tensor):
        return tf.map_fn(odeint_external, tensor, parallel_iterations=99)

    # result in (x, t)
    result = parallelized_func((x, t))

    if is_only_one_flag:
        return result[0][0]
    else:
        return result[0]
