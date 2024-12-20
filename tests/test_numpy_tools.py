import pytest

import astropy.units as u
import keras
import numpy as np
import numpy.testing as npt
import astroNN.nn.numpy
from astroNN.config import MAGIC_NUMBER


test_data = [np.array([-1.0, 2.0, 3.0, 4.0]), [-1.0, 2.0, 3.0, 4.0], 0.0]


@pytest.mark.parametrize("x", test_data)
def test_sigmoid(x):
    # make sure its the same as keras implementation
    np_x = astroNN.nn.numpy.sigmoid(x)
    keras_x = keras.ops.sigmoid(keras.ops.array(x, dtype="float32"))
    npt.assert_array_almost_equal(keras.ops.convert_to_numpy(keras_x), np_x)

    # make sure identity transform
    npt.assert_array_almost_equal(
        astroNN.nn.numpy.sigmoid_inv(astroNN.nn.numpy.sigmoid(x)), x
    )


@pytest.mark.parametrize("x", test_data)
def test_relu(x):
    # make sure its the same as keras implementation
    np_x = astroNN.nn.numpy.relu(x)
    keras_x = keras.ops.relu(keras.ops.array(x, dtype="float32"))
    npt.assert_array_equal(keras.ops.convert_to_numpy(keras_x), np_x)


@pytest.mark.parametrize("x", test_data)
def test_regularizator(x):
    reg = 0.2

    np_x_l1 = astroNN.nn.numpy.l1(x, l1=reg)
    np_x_l2 = astroNN.nn.numpy.l2(x, l2=reg)

    l1_reg = keras.regularizers.L1(l1=reg)
    l2_reg = keras.regularizers.L2(l2=reg)
    keras_x_l1 = l1_reg(keras.ops.array(x, dtype="float32"))
    keras_x_l2 = l2_reg(keras.ops.array(x, dtype="float32"))

    npt.assert_array_almost_equal(keras.ops.convert_to_numpy(keras_x_l1), np_x_l1)
    npt.assert_array_almost_equal(keras.ops.convert_to_numpy(keras_x_l2), np_x_l2)


def test_kl_divergence():
    x = np.random.normal(10, 0.5, 1000)

    # assert two equal vectors have 0 KL-Divergence
    npt.assert_equal(astroNN.nn.numpy.kl_divergence(x.tolist(), x.tolist()), 0.0)


def test_numpy_metrics():
    x = np.array([-2.0, 2.0])
    y = np.array([MAGIC_NUMBER, 4.0])

    # ------------------- Mean ------------------- #
    mape = astroNN.nn.numpy.mean_absolute_percentage_error(x * u.kpc, y * u.kpc)
    mape_unitless = astroNN.nn.numpy.mean_absolute_percentage_error(x, y)
    npt.assert_array_equal(mape, 50.0)
    npt.assert_array_equal(mape, mape_unitless)
    # assert error raise if only x or y carries astropy units
    with pytest.raises(TypeError):
        astroNN.nn.numpy.mean_absolute_percentage_error(x * u.kpc, y)
    with pytest.raises(TypeError):
        astroNN.nn.numpy.mean_absolute_percentage_error(x, y * u.kpc)

    mae = astroNN.nn.numpy.mean_absolute_error(x * u.kpc, y * u.kpc)
    mae_diffunits = (
        astroNN.nn.numpy.mean_absolute_error((x * u.kpc).to(u.parsec), y * u.kpc) / 1000
    )
    mae_ubnitless = astroNN.nn.numpy.mean_absolute_error(x, y)
    npt.assert_array_equal(mae, 2.0)
    npt.assert_array_equal(mae, mae_ubnitless)
    npt.assert_array_equal(mae, mae_diffunits)
    # assert error raise if only x or y carries astropy units
    with pytest.raises(TypeError):
        astroNN.nn.numpy.mean_absolute_error(x * u.kpc, y)
    with pytest.raises(TypeError):
        astroNN.nn.numpy.mean_absolute_error(x, y * u.kpc)

    # ------------------- Median ------------------- #
    npt.assert_equal(
        astroNN.nn.numpy.median_absolute_percentage_error(
            [2.0, 3.0, 7.0], [2.0, 1.0, 7.0]
        ),
        0.0,
    )
    npt.assert_equal(
        astroNN.nn.numpy.median_absolute_error([2.0, 3.0, 7.0], [2.0, 1.0, 7.0]), 0.0
    )
