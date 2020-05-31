import unittest

import numpy as np
import tensorflow as tf
import numpy.testing as npt
from astroNN.neuralode import odeint


class IntegratorTestCase(unittest.TestCase):
    def test_simpleODE(self):
        t = np.linspace(0, 10, 100)
        # initial condition
        true_y0 = [0., 5.]

        methods = ['dop853']
        true_func = lambda y, t: np.sin(5*t)
        ode_func = lambda y, t: tf.stack([5*tf.cos(5*t), -25*tf.sin(5*t)])
        for method in methods:
            true_y = odeint(ode_func, true_y0, t, method=method, precision=tf.float64)
            npt.assert_array_almost_equal(true_y.numpy()[:, 0], true_func(true_y0, t))


if __name__ == '__main__':
    unittest.main()