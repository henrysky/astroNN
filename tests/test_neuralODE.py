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
        true_func = lambda y, t: np.sin(5.*t)
        ode_func = lambda y, t: tf.stack([5.*tf.cos(5.*t), -25.*tf.sin(5.*t)])
        for method in methods:
            true_y = odeint(ode_func, true_y0, t, method=method, precision=tf.float64)
            npt.assert_array_almost_equal(true_y.numpy()[:, 0], true_func(true_y0, t))

    def test_ODEbadprecision(self):  # make sure float32 is not enough for very precise integration
        t = tf.constant(np.linspace(0, 5, 500), dtype=tf.float32)
        # initial condition
        true_y0 = tf.constant([0., 5.], dtype=tf.float32)

        true_func = lambda y, t: np.sin(5.*t)
        ode_func = lambda y, t: tf.stack([5.*tf.cos(5.*t), -25.*tf.sin(5.*t)])
        true_y = odeint(ode_func, true_y0, t, method='rk4', precision=tf.float32)
        np.testing.assert_array_almost_equal(true_y.numpy()[:, 0], true_func(true_y0, t), decimal=4)

        true_y0_pretend_multidims = [[0., 5.]]  # to introduce a mix of list, np array, tensor to make sure no issue
        true_y_pretend_multidims = odeint(ode_func, true_y0_pretend_multidims, t, method='rk4', precision=tf.float32)

        # assert equal pretendinging multidim or not, only need to check the last few elements beause its integration
        np.testing.assert_array_almost_equal(true_y_pretend_multidims[0].numpy()[400:, 0], true_y.numpy()[400:, 0], decimal=4)

        true_y0_multidims = tf.constant([[1., 2.], [0., 5.]], dtype=tf.float32)
        true_y_multidims = odeint(ode_func, true_y0_multidims, t, method='rk4', precision=tf.float32)

        # assert equal in multidim or not
        np.testing.assert_array_almost_equal(true_y_multidims[1].numpy()[:, 0], true_y.numpy()[:, 0], decimal=4)


if __name__ == '__main__':
    unittest.main()
