# ---------------------------------------------------------------#
#   astroNN.nn.numpy: tools written with numpy instead of tf
# ---------------------------------------------------------------#
import astropy.units as u
import numpy as np

from astroNN.config import MAGIC_NUMBER


def sigmoid(x):
    """
    NumPy implementation of tf.sigmoid, mask ``magicnumber``

    :param x: Data to be applied sigmoid activation
    :type x: Union[ndarray, float]
    :return: Sigmoid activated data
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    x = np.ma.array(x, mask=(x == MAGIC_NUMBER))
    return np.ma.divide(1, np.ma.add(1, np.divide(1, np.ma.exp(x))))


def sigmoid_inv(x):
    """
    NumPy implementation of tf.sigmoid inverse, mask ``magicnumber``

    :param x: Data to be applied inverse sigmoid activation
    :type x: Union[numpy.ndarray, float]
    :return: Inverse Sigmoid activated data
    :rtype: Union[numpy.ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    x = np.ma.array(x, mask=(x == MAGIC_NUMBER))
    return np.ma.log(np.ma.divide(x, np.ma.subtract(1, x)))


def l1(x, l1=0.0):
    """
    NumPy implementation of tf.keras.regularizers.l1

    :param x: Data to have L1 regularization coefficient calculated
    :type x: Union[ndarray, float]
    :param l1: L1 regularization parameter
    :type l1: Union[ndarray, float]
    :return: L1 regularization coefficient
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    l1_x = 0.0
    l1_x += np.sum(l1 * np.abs(x))
    return l1_x


def l2(x, l2=0.0):
    """
    NumPy implementation of tf.keras.regularizers.l2

    :param x: Data to have L2 regularization coefficient calculated
    :type x: Union[ndarray, float]
    :param l2: L2 regularization parameter
    :type l2: Union[ndarray, float]
    :return: L2 regularization coefficient
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    l2_x = 0.0
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


def mape_core(x, y, axis=None, mode=None):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(x, u.Quantity) and isinstance(y, u.Quantity):
        percentage = ((x - y) / y).value
        # still need to take the value for creating mask
        x = x.value
        y = y.value
    elif (isinstance(x, u.Quantity) and not isinstance(y, u.Quantity)) or (
        not isinstance(x, u.Quantity) and isinstance(y, u.Quantity)
    ):
        raise TypeError(
            "Only one of your data provided has astropy units \n"
            "Either both x and y are ndarray or both x and y are astropy.Quatity, "
            "return without astropy units in all case"
        )
    else:
        percentage = (x - y) / y
    if mode == "mean":
        return np.ma.mean(
            np.ma.array(
                np.abs(percentage) * 100.0,
                mask=((x == MAGIC_NUMBER) | (y == MAGIC_NUMBER)),
            ),
            axis=axis,
        )
    elif mode == "median":
        return np.ma.median(
            np.ma.array(
                np.abs(percentage) * 100.0,
                mask=[(x == MAGIC_NUMBER) | (y == MAGIC_NUMBER)],
            ),
            axis=axis,
        )


def mean_absolute_percentage_error(x, y, axis=None):
    """
    | NumPy implementation of tf.keras.metrics.mean_absolute_percentage_error with capability to deal with ``magicnumber``
      and astropy Quantity
    | Either both x and y are ndarray or both x and y are astropy.Quatity, return has no astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float, astropy.Quatity]
    :param y: ground truth
    :type y: Union[ndarray, float, astropy.Quatity]
    :param axis: NumPy axis
    :type axis: Union[NoneType, int]
    :raise: TypeError when only either x or y contains astropy units. Both x, y should carry/not carry astropy units at the same time
    :return: Mean Absolute Percentage Error
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return mape_core(x, y, axis=axis, mode="mean")


def median_absolute_percentage_error(x, y, axis=None):
    """
    | NumPy implementation of a median version of tf.keras.metrics.mean_absolute_percentage_error with capability to
    | deal with ``magicnumber`` and astropy Quantity
    | Either both x and y are ndarray or both x and y are astropy.Quatity, return has no astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float, astropy.Quatity]
    :param y: ground truth
    :type y: Union[ndarray, float, astropy.Quatity]
    :param axis: NumPy axis
    :type axis: Union[NoneType, int]
    :raise: TypeError when only either x or y contains astropy units. Both x, y should carry/not carry astropy units at the same time
    :return: Median Absolute Percentage Error
    :rtype: Union[ndarray, float]
    :History: 2018-May-13 - Written - Henry Leung (University of Toronto)
    """
    return mape_core(x, y, axis=axis, mode="median")


def mae_core(x, y, axis=None, mode=None):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(x, u.Quantity) and isinstance(y, u.Quantity):
        diff = (x - y).value
        # still need to take the value for creating mask
        x = x.value
        y = y.value
    elif (isinstance(x, u.Quantity) and not isinstance(y, u.Quantity)) or (
        not isinstance(x, u.Quantity) and isinstance(y, u.Quantity)
    ):
        raise TypeError(
            "Only one of your data provided has astropy units \n"
            "Either both x and y are ndarray or both x and y are astropy.Quatity, "
            "return without astropy units in all case"
        )
    else:
        diff = x - y
    if mode == "mean":
        return np.ma.mean(
            np.ma.array(np.abs(diff), mask=((x == MAGIC_NUMBER) | (y == MAGIC_NUMBER))),
            axis=axis,
        )
    elif mode == "median":
        return np.ma.median(
            np.ma.array(np.abs(diff), mask=[(x == MAGIC_NUMBER) | (y == MAGIC_NUMBER)]),
            axis=axis,
        )


def mean_absolute_error(x, y, axis=None):
    """
    NumPy implementation of tf.keras.metrics.mean_absolute_error  with capability to deal with ``magicnumber``
    and astropy Quantity

    Either both x and y are ndarray or both x and y are astropy.Quatity, return without astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float, astropy.Quatity]
    :param y: ground truth
    :type y: Union[ndarray, float, astropy.Quatity]
    :param axis: NumPy axis
    :type axis: Union[NoneType, int]
    :raise: TypeError when only either x or y contains astropy units. Both x, y should carry/not carry astropy units at the same time
    :return: Mean Absolute Error
    :rtype: Union[ndarray, float]
    :History: 2018-Apr-11 - Written - Henry Leung (University of Toronto)
    """
    return mae_core(x, y, axis=axis, mode="mean")


def median_absolute_error(x, y, axis=None):
    """
    NumPy implementation of a median version of tf.keras.metrics.mean_absolute_error  with capability to deal with
    ``magicnumber`` and astropy Quantity

    Either both x and y are ndarray or both x and y are astropy.Quatity, return without astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float, astropy.Quatity]
    :param y: ground truth
    :type y: Union[ndarray, float, astropy.Quatity]
    :param axis: NumPy axis
    :type axis: Union[NoneType, int]
    :raise: TypeError when only either x or y contains astropy units. Both x, y should carry/not carry astropy units at the same time
    :return: Median Absolute Error
    :rtype: Union[ndarray, float]
    :History: 2018-May-13 - Written - Henry Leung (University of Toronto)
    """
    return mae_core(x, y, axis=axis, mode="median")


def kl_divergence(x, y):
    """
    NumPy implementation of tf.distributions.kl_divergence

    Either both x and y are ndarray or both x and y are astropy.Quatity, return without astropy units in all case

    :param x: prediction
    :type x: Union[ndarray, float]
    :param y: ground truth
    :type y: Union[ndarray, float]
    :return: KL-divergence
    :rtype: Union[ndarray, float]
    :History: 2018-May-13 - Written - Henry Leung (University of Toronto)
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    x = np.clip(x, 1e-10, 1)
    y = np.clip(y, 1e-10, 1)
    return np.sum(x * np.log(x / y), axis=-1)
