import unittest
import numpy.testing as npt

import tensorflow as tf
import keras.backend as K

from astroNN import MAGIC_NUMBER
from astroNN.nn.losses import mean_absolute_error, mean_squared_error

from keras.losses import mse

class MyTestCase(unittest.TestCase):
    def test_loss_func(self):
        y_pred = tf.Variable([2., 3., 4.])
        y_true = tf.Variable([2., MAGIC_NUMBER, 4.])

        # make sure loss functions handle magic_number correctly
        self.assertEqual(mean_absolute_error(y_true, y_pred).eval(session=K.get_session()), 0.)
        self.assertEqual(mean_squared_error(y_true, y_pred).eval(session=K.get_session()), 0.)

        # =============multi dimensional case============= #
        y_pred = tf.Variable([[2., 3., 4.], [2., 3., 7.]])
        y_true = tf.Variable([[2., MAGIC_NUMBER, 4.], [2., MAGIC_NUMBER, 4.]])
        npt.assert_array_equal(mean_absolute_error(y_true, y_pred).eval(session=K.get_session()), [0., 3. / 3.])
        npt.assert_array_equal(mean_squared_error(y_true, y_pred).eval(session=K.get_session()), [0., 9. / 3])


if __name__ == '__main__':
    unittest.main()
