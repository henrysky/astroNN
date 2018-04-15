import unittest

import astropy.units as u
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from astroNN.config import keras_import_manager
from astroNN.nn.numpy import sigmoid, sigmoid_inv, relu, l1, l2

keras = keras_import_manager()
get_session = keras.backend.get_session

# force the test to use CPU, using GPU will be much slower for such small test
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
keras.backend.set_session(sess)


class MyTestCase(unittest.TestCase):
    def test_sigmoid(self):
        # make sure its the same as tensorflow
        x = np.array([-1., 2., 3., 4.])
        tf_x = tf.nn.sigmoid(tf.convert_to_tensor(x))
        astroNN_x = sigmoid(x)
        npt.assert_array_equal(tf_x.eval(session=get_session()), astroNN_x)

        # make sure identity transform
        npt.assert_array_almost_equal(sigmoid_inv(sigmoid(x)), x)

    def test_relu(self):
        # make sure its the same as tensorflow
        x = np.array([-1., 2., 3., 4.])
        tf_x = tf.nn.relu(tf.convert_to_tensor(x))
        astroNN_x = relu(x)
        npt.assert_array_equal(tf_x.eval(session=get_session()), astroNN_x)

    def test_regularizator(self):
        # make sure its the same as tensorflow
        x = np.array([-1., 2., 3., 4.])
        reg = 0.2
        l1_reg = tf.keras.regularizers.l1(l=reg)
        l2_reg = tf.keras.regularizers.l2(l=reg)

        tf_x = l1_reg(tf.convert_to_tensor(x))
        tf_x_2 = l2_reg(tf.convert_to_tensor(x))

        astroNN_x = l1(x, l1=reg)
        astroNN_x_2 = l2(x, l2=reg)

        npt.assert_array_almost_equal(tf_x.eval(session=get_session()), astroNN_x)
        npt.assert_array_almost_equal(tf_x_2.eval(session=get_session()), astroNN_x_2)

    def test_numpy_metrics(self):
        from astroNN.nn.numpy import mean_absolute_percentage_error
        x = np.array([-2., 2.])
        y = np.array([-9999., 4.])

        mape = mean_absolute_percentage_error(x * u.kpc, y * u.kpc)
        mape_ubnitless = mean_absolute_percentage_error(x, y)
        npt.assert_array_equal(mape, 0.5)
        npt.assert_array_equal(mape, mape_ubnitless)


if __name__ == '__main__':
    unittest.main()
