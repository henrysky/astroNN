import keras
import numpy as np
import numpy.testing as npt
from astroNN.shared.nn_tools import cpu_fallback
from astroNN.config import MAGIC_NUMBER
from astroNN.nn.losses import (
    magic_correction_term,
    mean_absolute_error,
    mean_squared_error,
    categorical_crossentropy,
    binary_crossentropy,
    nll,
    mean_error,
    zeros_loss,
    mean_percentage_error,
    median,
)
from astroNN.nn.metrics import (
    categorical_accuracy,
    binary_accuracy,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    median_error,
    median_absolute_deviation,
    mad_std,
)

# make sure this test use CPU
cpu_fallback()


def test_loss_magic():
    # =============Magic correction term============= #
    y_true = keras.ops.array(
        [[2.0, MAGIC_NUMBER, MAGIC_NUMBER], [2.0, MAGIC_NUMBER, 4.0]]
    )
    npt.assert_array_equal(
        keras.ops.convert_to_numpy(magic_correction_term(y_true)), [3.0, 1.5]
    )


def test_loss_mse():
    # =============MSE/MAE============= #
    y_pred = keras.ops.array([[2.0, 3.0, 4.0], [2.0, 3.0, 7.0]])
    y_pred_2 = keras.ops.array([[2.0, 9.0, 4.0], [2.0, 0.0, 7.0]])
    y_true = keras.ops.array([[2.0, MAGIC_NUMBER, 4.0], [2.0, MAGIC_NUMBER, 4.0]])

    npt.assert_almost_equal(
        keras.ops.convert_to_numpy(mean_absolute_error(y_true, y_pred)),
        [0.0, 3.0 / 2.0],
    )
    npt.assert_almost_equal(
        keras.ops.convert_to_numpy(mean_squared_error(y_true, y_pred)), [0.0, 9.0 / 2]
    )

    # make sure neural network prediction won't matter for magic number term
    npt.assert_almost_equal(
        keras.ops.convert_to_numpy(mean_absolute_error(y_true, y_pred)),
        keras.ops.convert_to_numpy(mean_absolute_error(y_true, y_pred_2)),
    )
    npt.assert_almost_equal(
        keras.ops.convert_to_numpy(mean_squared_error(y_true, y_pred)),
        keras.ops.convert_to_numpy(mean_squared_error(y_true, y_pred_2)),
    )


def test_loss_mean_err():
    # =============Mean Error============= #
    y_pred = keras.ops.array([[1.0, 3.0, 4.0], [2.0, 3.0, 7.0]])
    y_true = keras.ops.array([[2.0, MAGIC_NUMBER, 3.0], [2.0, MAGIC_NUMBER, 7.0]])

    npt.assert_almost_equal(
        keras.ops.convert_to_numpy(mean_error(y_true, y_pred)), [0.0, 0.0]
    )


def test_loss_acurrancy():
    # =============Accuracy============= #
    y_pred = keras.ops.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [0.0, MAGIC_NUMBER, 1.0]])

    npt.assert_array_equal(
        keras.ops.convert_to_numpy(categorical_accuracy(y_true, y_pred)), [1.0, 0.0]
    )
    npt.assert_almost_equal(
        keras.ops.convert_to_numpy(binary_accuracy(y_true, y_pred)), [1.0 / 2.0, 0.0]
    )


def test_loss_abs_error():
    # =============Abs Percentage Accuracy============= #
    y_pred = keras.ops.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred_2 = keras.ops.array([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(mean_absolute_percentage_error(y_true, y_pred)),
        [50.0, 50.0],
        decimal=3,
    )
    # make sure neural network prediction won't matter for magic number term
    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(mean_absolute_percentage_error(y_true, y_pred)),
        keras.ops.convert_to_numpy(mean_absolute_percentage_error(y_true, y_pred_2)),
        decimal=3,
    )


def test_loss_percentage_error():
    # =============Percentage Accuracy============= #
    y_pred = keras.ops.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred_2 = keras.ops.array([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(mean_percentage_error(y_true, y_pred)),
        [50.0, 50.0],
        decimal=3,
    )
    # make sure neural network prediction won't matter for magic number term
    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(mean_percentage_error(y_true, y_pred)),
        keras.ops.convert_to_numpy(mean_percentage_error(y_true, y_pred_2)),
        decimal=3,
    )


def test_loss_log_error():
    # =============Mean Squared Log Error============= #
    y_pred = keras.ops.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred_2 = keras.ops.array([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(mean_squared_logarithmic_error(y_true, y_pred)),
        [0.24, 0.24],
        decimal=3,
    )
    # make sure neural network prediction won't matter for magic number term
    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(mean_squared_logarithmic_error(y_true, y_pred)),
        keras.ops.convert_to_numpy(mean_squared_logarithmic_error(y_true, y_pred_2)),
        decimal=3,
    )


def test_loss_zeros():
    # =============Zeros Loss============= #
    y_pred = keras.ops.array([[1.0, 0.0, 0.0], [5.0, -9.0, 2.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(zeros_loss(y_true, y_pred)), [0.0, 0.0]
    )


def test_categorical_crossentropy():
    # Truth with Magic number is wrong
    y_pred = keras.ops.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 0.0]])

    y_pred_softmax = keras.ops.softmax(y_pred)

    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(categorical_crossentropy(y_true, y_pred_softmax)),
        keras.ops.convert_to_numpy(
            categorical_crossentropy(y_true, y_pred, from_logits=True)
        ),
        decimal=3,
    )


def test_binary_crossentropy():
    y_pred = keras.ops.array([[0.5, 0.0, 1.0], [2.0, 0.0, -1.0]])
    y_pred_2 = keras.ops.array([[0.5, 2.0, 1.0], [2.0, 2.0, -1.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 0.0]])
    y_pred_sigmoid = keras.ops.sigmoid(y_pred)
    y_pred_2_sigmoid = keras.ops.sigmoid(y_pred_2)

    # Truth with Magic number is wrong
    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(binary_crossentropy(y_true, y_pred_sigmoid)),
        keras.ops.convert_to_numpy(
            binary_crossentropy(y_true, y_pred, from_logits=True)
        ),
        decimal=3,
    )
    # make sure neural network prediction won't matter for magic number term
    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(
            binary_crossentropy(y_true, y_pred_2, from_logits=True)
        ),
        keras.ops.convert_to_numpy(
            binary_crossentropy(y_true, y_pred, from_logits=True)
        ),
        decimal=3,
    )
    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(binary_crossentropy(y_true, y_pred_sigmoid)),
        keras.ops.convert_to_numpy(binary_crossentropy(y_true, y_pred_2_sigmoid)),
        decimal=3,
    )


def test_negative_log_likelihood():
    y_pred = keras.ops.array([[0.5, 0.0, 1.0], [2.0, 0.0, -1.0]])
    y_true = keras.ops.array([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 0.0]])

    npt.assert_array_almost_equal(
        keras.ops.convert_to_numpy(nll(y_true, y_pred)), 0.34657377, decimal=3
    )


def test_median():
    y_pred = keras.ops.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    npt.assert_array_almost_equal(median(y_pred), np.median(y_pred), decimal=3)
    npt.assert_array_almost_equal(
        median(y_pred, axis=1), np.median(y_pred, axis=1), decimal=3
    )
    npt.assert_array_almost_equal(
        median(y_pred, axis=0), np.median(y_pred, axis=0), decimal=3
    )


def test_mad_std():
    test_array = np.random.normal(0.0, 1.0, 100000)
    npt.assert_equal(
        np.round(
            keras.ops.convert_to_numpy(
                mad_std(test_array, np.zeros_like(test_array), axis=None)
            )
        ),
        1.0,
    )


def test_median_metrics():
    y_pred = keras.ops.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y_true = keras.ops.array([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])

    npt.assert_array_almost_equal(
        median_error(y_true, y_pred, axis=None), np.median(y_true - y_pred), decimal=3
    )
    npt.assert_array_almost_equal(
        median_absolute_deviation(y_true, y_pred, axis=None),
        np.median(np.abs(y_true - y_pred)),
        decimal=3,
    )
