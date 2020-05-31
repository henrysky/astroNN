import tensorflow as tf
from astroNN.neuralode.dop853 import dop853


def odeint(func=None, x=None, t=None, method='dop853', *args, **kwargs):
    """
    To computes the numerical solution of a system of first order ordinary differential equations y'=f(x,y).

    :param func: function of the differential equation, usually take func([position, velocity], time) and return velocity, acceleration
    :type func: callable
    :param x: initial x, usually is [position, velocity]
    :type x: Union([tf.Tensor, numpy.ndarray, list])
    :param t: set of times at which one wants the result
    :type t: Union([tf.Tensor, numpy.ndarray, list])

    :return: integrated result
    :rtype: tf.Tensor

    :History: 2018-Nov-23 - Written - Henry Leung (University of Toronto)
    """
    if method.lower() == 'dop853':
        ode_method = dop853
    else:
        raise NotImplementedError(f"Method {method} is not implemented")

    # check things if they are tensors
    if not isinstance(x, tf.Tensor):
        x = tf.constant(x)
    if not isinstance(x, tf.Tensor):
        x = tf.constant(x)

    return ode_method(func=func, x=x, t=t, *args, **kwargs)
