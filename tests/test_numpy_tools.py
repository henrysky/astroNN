import unittest

import astropy.units as u
import numpy as np
import numpy.testing as npt
import tensorflow as tf
from tensorflow.python import context

from astroNN.config import MAGIC_NUMBER
from astroNN.nn.numpy import mean_absolute_percentage_error, mean_absolute_error, median_absolute_error, \
    median_absolute_percentage_error, kl_divergence
from astroNN.nn.numpy import sigmoid, sigmoid_inv, relu, l1, l2


# noinspection PyUnresolvedReferences
class MyTestCase(unittest.TestCase):
    def test_sigmoid(self):
        # make sure its the same as tensorflow
        x = np.array([-1., 2., 3., 4.])
        astroNN_x = sigmoid(x)
        tf_x = tf.nn.sigmoid(x)
        npt.assert_array_almost_equal(tf_x.numpy(), astroNN_x)

        # make sure identity transform
        npt.assert_array_almost_equal(sigmoid_inv(sigmoid(x)), x)

        # for list
        # make sure its the same as tensorflow
        x = [-1., 2., 3., 4.]
        astroNN_x_list = sigmoid(x)
        npt.assert_array_almost_equal(astroNN_x_list, astroNN_x)

        # make sure identity transform
        npt.assert_array_almost_equal(sigmoid_inv(sigmoid(x)), x)

        # for float
        # make sure its the same as tensorflow
        x = 0.
        astroNN_x = sigmoid(x)
        npt.assert_array_equal(0.5, astroNN_x)

        # make sure identity transform
        npt.assert_array_almost_equal(sigmoid_inv(sigmoid(x)), x)

    def test_relu(self):
        # make sure its the same as tensorflow
        x = np.array([-1., 2., 3., 4.])
        astroNN_x = relu(x)
        tf_x = tf.nn.relu(x)
        npt.assert_array_equal(tf_x.numpy(), astroNN_x)

    def test_kl_divergence(self):
        x = np.random.normal(10, 0.5, 1000)

        # assert two equal vectors have 0 KL-Divergence
        self.assertEqual(kl_divergence(x.tolist(), x.tolist()), 0.)

    def test_regularizator(self):
        # make sure its the same as tensorflow
        x = np.array([-1., 2., 3., 4.])
        reg = 0.2

        astroNN_x = l1(x, l1=reg)
        astroNN_x_2 = l2(x, l2=reg)

        l1_reg = tf.keras.regularizers.l1(l=reg)
        l2_reg = tf.keras.regularizers.l2(l=reg)
        tf_x = l1_reg(x)
        tf_x_2 = l2_reg(x)

        npt.assert_array_almost_equal(tf_x.numpy(), astroNN_x)
        npt.assert_array_almost_equal(tf_x_2.numpy(), astroNN_x_2)

    def test_numpy_metrics(self):
        x = np.array([-2., 2.])
        y = np.array([MAGIC_NUMBER, 4.])

        # ------------------- Mean ------------------- #
        mean_absolute_error([2., 3., 7.], [2., 0., 7.])
        mape = mean_absolute_percentage_error(x * u.kpc, y * u.kpc)
        mape_ubnitless = mean_absolute_percentage_error(x, y)
        npt.assert_array_equal(mape, 50.)
        npt.assert_array_equal(mape, mape_ubnitless)
        # assert error raise if only x or y carries astropy units
        self.assertRaises(TypeError, mean_absolute_percentage_error, x * u.kpc, y)
        self.assertRaises(TypeError, mean_absolute_percentage_error, x, y * u.kpc)

        mae = mean_absolute_error(x * u.kpc, y * u.kpc)
        mae_diffunits = mean_absolute_error((x * u.kpc).to(u.parsec), y * u.kpc) / 1000
        mae_ubnitless = mean_absolute_error(x, y)
        npt.assert_array_equal(mae, 2.)
        npt.assert_array_equal(mae, mae_ubnitless)
        npt.assert_array_equal(mae, mae_diffunits)
        # assert error raise if only x or y carries astropy units
        self.assertRaises(TypeError, mean_absolute_error, x * u.kpc, y)
        self.assertRaises(TypeError, mean_absolute_error, x, y * u.kpc)

        # ------------------- Median ------------------- #

        self.assertEqual(median_absolute_percentage_error([2., 3., 7.], [2., 1., 7.]), 0.)
        self.assertEqual(median_absolute_error([2., 3., 7.], [2., 1., 7.]), 0.)


if __name__ == '__main__':
    unittest.main()
