import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from astroNN.shared.nn_tools import cpu_fallback

# make sure this test use CPU
cpu_fallback()

from astroNN.config import MAGIC_NUMBER
from astroNN.nn import reduce_var
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
    median
)
from astroNN.nn.metrics import (
    categorical_accuracy,
    binary_accuracy,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    median_error,
    median_absolute_deviation,
    mad_std
)


class LossFuncTestCase(unittest.TestCase):
    def test_loss_func_util(self):
        # make sure custom reduce_var works
        content = [1, 2, 3, 4, 5]
        var_array = tf.constant(content)
        self.assertEqual(reduce_var(var_array).numpy(), np.var(content))

    def test_loss_magic(self):
        # =============Magic correction term============= #
        y_true = tf.constant(
            [[2.0, MAGIC_NUMBER, MAGIC_NUMBER], [2.0, MAGIC_NUMBER, 4.0]]
        )
        npt.assert_array_equal(magic_correction_term(y_true).numpy(), [3.0, 1.5])

    def test_loss_mse(self):
        # =============MSE/MAE============= #
        y_pred = tf.constant([[2.0, 3.0, 4.0], [2.0, 3.0, 7.0]])
        y_pred_2 = tf.constant([[2.0, 9.0, 4.0], [2.0, 0.0, 7.0]])
        y_true = tf.constant([[2.0, MAGIC_NUMBER, 4.0], [2.0, MAGIC_NUMBER, 4.0]])

        npt.assert_almost_equal(
            mean_absolute_error(y_true, y_pred).numpy(), [0.0, 3.0 / 2.0]
        )
        npt.assert_almost_equal(
            mean_squared_error(y_true, y_pred).numpy(), [0.0, 9.0 / 2]
        )

        # make sure neural network prediction won't matter for magic number term
        npt.assert_almost_equal(
            mean_absolute_error(y_true, y_pred).numpy(),
            mean_absolute_error(y_true, y_pred_2).numpy(),
        )
        npt.assert_almost_equal(
            mean_squared_error(y_true, y_pred).numpy(),
            mean_squared_error(y_true, y_pred_2).numpy(),
        )

    def test_loss_mean_err(self):
        # =============Mean Error============= #
        y_pred = tf.constant([[1.0, 3.0, 4.0], [2.0, 3.0, 7.0]])
        y_true = tf.constant([[2.0, MAGIC_NUMBER, 3.0], [2.0, MAGIC_NUMBER, 7.0]])

        npt.assert_almost_equal(mean_error(y_true, y_pred).numpy(), [0.0, 0.0])

    def test_loss_acurrancy(self):
        # =============Accuracy============= #
        y_pred = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [0.0, MAGIC_NUMBER, 1.0]])

        npt.assert_array_equal(categorical_accuracy(y_true, y_pred).numpy(), [1.0, 0.0])
        npt.assert_almost_equal(
            binary_accuracy(y_true, y_pred).numpy(), [1.0 / 2.0, 0.0]
        )

    def test_loss_abs_error(self):
        # =============Abs Percentage Accuracy============= #
        y_pred = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_pred_2 = tf.constant([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

        npt.assert_array_almost_equal(
            mean_absolute_percentage_error(y_true, y_pred).numpy(),
            [50.0, 50.0],
            decimal=3,
        )
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(
            mean_absolute_percentage_error(y_true, y_pred).numpy(),
            mean_absolute_percentage_error(y_true, y_pred_2).numpy(),
            decimal=3,
        )

    def test_loss_percentage_error(self):
        # =============Percentage Accuracy============= #
        y_pred = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_pred_2 = tf.constant([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

        npt.assert_array_almost_equal(
            mean_percentage_error(y_true, y_pred).numpy(), [50.0, 50.0], decimal=3
        )
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(
            mean_percentage_error(y_true, y_pred).numpy(),
            mean_percentage_error(y_true, y_pred_2).numpy(),
            decimal=3,
        )

    def test_loss_log_error(self):
        # =============Mean Squared Log Error============= #
        y_pred = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_pred_2 = tf.constant([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

        npt.assert_array_almost_equal(
            mean_squared_logarithmic_error(y_true, y_pred).numpy(),
            [0.24, 0.24],
            decimal=3,
        )
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(
            mean_squared_logarithmic_error(y_true, y_pred).numpy(),
            mean_squared_logarithmic_error(y_true, y_pred_2).numpy(),
            decimal=3,
        )

    def test_loss_zeros(self):
        # =============Zeros Loss============= #
        y_pred = tf.constant([[1.0, 0.0, 0.0], [5.0, -9.0, 2.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 1.0]])

        npt.assert_array_almost_equal(zeros_loss(y_true, y_pred).numpy(), [0.0, 0.0])

    def test_categorical_crossentropy(self):
        # Truth with Magic number is wrong
        y_pred = tf.constant([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 0.0]])

        y_pred_softmax = tf.nn.softmax(y_pred)

        npt.assert_array_almost_equal(
            categorical_crossentropy(y_true, y_pred_softmax).numpy(),
            categorical_crossentropy(y_true, y_pred, from_logits=True).numpy(),
            decimal=3,
        )

    def test_binary_crossentropy(self):
        y_pred = tf.constant([[0.5, 0.0, 1.0], [2.0, 0.0, -1.0]])
        y_pred_2 = tf.constant([[0.5, 2.0, 1.0], [2.0, 2.0, -1.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 0.0]])
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        y_pred_2_sigmoid = tf.nn.sigmoid(y_pred_2)

        # Truth with Magic number is wrong
        npt.assert_array_almost_equal(
            binary_crossentropy(y_true, y_pred_sigmoid).numpy(),
            binary_crossentropy(y_true, y_pred, from_logits=True).numpy(),
            decimal=3,
        )
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(
            binary_crossentropy(y_true, y_pred_2, from_logits=True).numpy(),
            binary_crossentropy(y_true, y_pred, from_logits=True).numpy(),
            decimal=3,
        )
        npt.assert_array_almost_equal(
            binary_crossentropy(y_true, y_pred_sigmoid).numpy(),
            binary_crossentropy(y_true, y_pred_2_sigmoid).numpy(),
            decimal=3,
        )

    def test_negative_log_likelihood(self):
        y_pred = tf.constant([[0.5, 0.0, 1.0], [2.0, 0.0, -1.0]])
        y_true = tf.constant([[1.0, MAGIC_NUMBER, 1.0], [1.0, MAGIC_NUMBER, 0.0]])

        npt.assert_array_almost_equal(
            nll(y_true, y_pred).numpy(), 0.34657377, decimal=3
        )
        
    def test_median(self):
        y_pred = tf.constant([[1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]])
        npt.assert_array_almost_equal(median(y_pred), np.median(y_pred), decimal=3)
        npt.assert_array_almost_equal(median(y_pred, axis=1), np.median(y_pred, axis=1), decimal=3)
        npt.assert_array_almost_equal(median(y_pred, axis=0), np.median(y_pred, axis=0), decimal=3)
        
    def test_mad_std(self):
        test_array = np.random.normal(0., 1., 100000)
        self.assertEqual(np.round(mad_std(test_array, np.zeros_like(test_array), axis=None)), 1.)
        
    def test_median_metrics(self):
        y_pred = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_true = tf.constant([[1.0, 9.0, 0.0], [1.0, -1.0, 0.0]])
        
        npt.assert_array_almost_equal(median_error(y_true, y_pred, axis=None), np.median(y_true - y_pred), decimal=3)
        npt.assert_array_almost_equal(median_absolute_deviation(y_true, y_pred, axis=None), np.median(np.abs(y_true - y_pred)), decimal=3)


if __name__ == "__main__":
    unittest.main()
