import unittest

import numpy as np
import tensorflow as tf
import numpy.testing as npt
from astroNN.neuralode import odeint


class IntegratorTestCase(unittest.TestCase):
    def test_simpleODE(self):
        t = tf.constant(np.linspace(0, 10, 1000), dtype=tf.float64)
        # initial condition
        true_y0 = tf.constant([0., 5.], dtype=tf.float64)

        methods = ['dop853', 'rk4']
        true_func = lambda y, t: np.sin(5*t)
        ode_func = lambda y, t: tf.cast(tf.stack([5*tf.cos(5*t), -25*tf.sin(5*t)]), tf.float64)
        for method in methods:
            true_y = odeint(ode_func, true_y0, t, method=method, precision=tf.float64)
            npt.assert_array_almost_equal(true_y.numpy()[:, 0], true_func(true_y0, t))

    def test_ODEbadprecision(self):  # make sure float32 is not enough for very precise integration
        t = tf.constant(np.linspace(0, 10, 1000), dtype=tf.float32)
        # initial condition
        true_y0 = tf.constant([0., 5.], dtype=tf.float32)

        true_func = lambda y, t: np.sin(5*t)
        ode_func = lambda y, t: tf.cast(tf.stack([5*tf.cos(5*t), -25*tf.sin(5*t)]), tf.float32)
        true_y = odeint(ode_func, true_y0, t, method='dop853', precision=tf.float32)
        self.assertRaises(AssertionError, npt.assert_array_almost_equal, true_y.numpy()[:, 0], true_func(true_y0, t))

        true_y0_pretend_multidims = [[0., 5.]]  # to introduce a mix of list, np array, tensor to make sure no issue
        true_y_pretend_multidims = odeint(ode_func, true_y0_pretend_multidims, t, method='dop853', precision=tf.float32)

        # assert equal pretendinging multidim or not
        np.testing.assert_array_almost_equal(true_y_pretend_multidims[0], true_y)

        true_y0_multidims = tf.constant([[1., 2.], [0., 5.]], dtype=tf.float32)
        t = np.linspace(0, 10, 1000)
        true_y_multidims = odeint(ode_func, true_y0_multidims, t, method='dop853', precision=tf.float32)

        # assert equal in multidim or not
        np.testing.assert_array_almost_equal(true_y_multidims[1], true_y)


if __name__ == '__main__':
    unittest.main()
