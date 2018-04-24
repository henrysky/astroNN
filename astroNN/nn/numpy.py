# ---------------------------------------------------------------#
#   astroNN.nn.numpy: tools written with numpy instead of tf
# ---------------------------------------------------------------#
import astropy.units as u
import numpy as np
from astroNN.config import MAGIC_NUMBER


def sigmoid(x):
    """
    NumPy implementation of tf.sigmoid

    :param x: Data to be applied sigmoid activation
    :type x: Union[ndarray, float]
    :return: Sigmoid activated data
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_inv(x):
    """
    NumPy implementation of tf.sigmoid inverse

    :param x: Data to be applied inverse sigmoid activation
    :type x: Union[numpy.ndarray, float]
    :return: Inverse Sigmoid activated data
    :rtype: Union[numpy.ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return np.log(x / (1 - x))


def l1(x, l1=0.):
    """
    NumPy implementation of tf.keras.regularizers.l1

    :param x: Data to have L1 regularization coefficient calculated
    :type x: Union[ndarray, float]
    :return: 1 regularization coefficient
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    l1_x = 0.
    l1_x += np.sum(l1 * np.abs(x))
    return l1_x


def l2(x, l2=0.):
    """
    NumPy implementation of tf.keras.regularizers.l2

    :param x: Data to have L2 regularization coefficient calculated
    :type x: Union[ndarray, float]
    :return: 1 regularization coefficient
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    l2_x = 0.
    l2_x += np.sum(l2 * np.square(x))
    return l2_x


def relu(x):
    """
    NumPy implementation of tf.nn.relu

    :param x: Data to have ReLU activated
    :type x: Union[ndarray, float]
    :return: ReLU activated data
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return x * (x > 0)


def mean_absolute_percentage_error(x, y, axis=None):
    """
    NumPy implementation of tf.keras.metrics.mean_absolute_percentage_error with capability to deal with ``magicnumber``
    and astropy Quantity

    Either both x and y are ndarray or both x and y are astropy.Quatity, return has no astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float, astropy.Quatity]
    :param y: ground truth
    :type y: Union[ndarray, float, astropy.Quatity]
    :param axis: NumPy axis
    :type axis: int
    :return: Mean Absolute Precentage Error
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    if type(x) == u.quantity.Quantity and type(y) == u.quantity.Quantity:
        percetnage = ((x - y) / y).value
        # still need to take the value for creating mask
        x = x.value
        y = y.value
    else:
        percetnage = (x - y) / y
    return np.ma.array(np.abs(percetnage) * 100., mask=[(x == MAGIC_NUMBER) | (y == MAGIC_NUMBER)]).mean(axis=axis)


def mean_absolute_error(x, y, axis=None):
    """
    NumPy implementation of tf.keras.metrics.mean_absolute_error  with capability to deal with ``magicnumber``
    and astropy Quantity

    Either both x and y are ndarray or both x and y are astropy.Quatity, return has no astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float, astropy.Quatity]
    :param y: ground truth
    :type y: Union[ndarray, float, astropy.Quatity]
    :param axis: NumPy axis
    :type axis: int
    :return: Mean Absolute Error
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    if type(x) == u.quantity.Quantity and type(y) == u.quantity.Quantity:
        diff = (x - y).value
        # still need to take the value for creating mask
        x = x.value
        y = y.value
    else:
        diff = (x - y)
    return np.ma.array(np.abs(diff), mask=[(x == MAGIC_NUMBER) | (y == MAGIC_NUMBER)]).mean(axis=axis)
