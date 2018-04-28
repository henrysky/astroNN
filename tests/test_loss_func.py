import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from astroNN.config import MAGIC_NUMBER, keras_import_manager
from astroNN.nn import magic_correction_term, reduce_var
from astroNN.nn.losses import mean_absolute_error, mean_squared_error, categorical_crossentropy, binary_crossentropy, \
    nll
from astroNN.nn.metrics import categorical_accuracy, binary_accuracy, mean_absolute_percentage_error, \
    mean_squared_logarithmic_error

keras = keras_import_manager()
get_session = keras.backend.get_session

# force the test to use CPU, using GPU will be much slower for such small test
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
keras.backend.set_session(sess)


class LossFuncTestCase(unittest.TestCase):
    def test_loss_func(self):
        # make sure custom reduce_var works
        var_array = [1, 2, 3, 4, 5]
        self.assertEqual(reduce_var(tf.Variable(var_array)).eval(session=get_session()), np.var(var_array))

        # =============Magic correction term============= #
        y_true = tf.Variable([[2., MAGIC_NUMBER, MAGIC_NUMBER], [2., MAGIC_NUMBER, 4.]])
        npt.assert_array_equal(magic_correction_term(y_true).eval(session=get_session()), [3., 1.5])

        # =============MSE/MAE============= #
        y_pred = tf.Variable([[2., 3., 4.], [2., 3., 7.]])
        y_pred_2 = tf.Variable([[2., 9., 4.], [2., 0., 7.]])
        y_true = tf.Variable([[2., MAGIC_NUMBER, 4.], [2., MAGIC_NUMBER, 4.]])
        npt.assert_almost_equal(mean_absolute_error(y_true, y_pred).eval(session=get_session()), [0., 3. / 2.])
        npt.assert_almost_equal(mean_squared_error(y_true, y_pred).eval(session=get_session()), [0., 9. / 2])

        # make sure neural network prediction won't matter for magic number term
        npt.assert_almost_equal(mean_absolute_error(y_true, y_pred).eval(session=get_session()),
                                mean_absolute_error(y_true, y_pred_2).eval(session=get_session()))
        npt.assert_almost_equal(mean_squared_error(y_true, y_pred).eval(session=get_session()),
                                mean_squared_error(y_true, y_pred_2).eval(session=get_session()))

        # =============Accuracy============= #
        y_pred = tf.Variable([[1., 0., 0.], [1., 0., 0.]])
        y_true = tf.Variable([[1., MAGIC_NUMBER, 1.], [0., MAGIC_NUMBER, 1.]])
        npt.assert_array_equal(categorical_accuracy(y_true, y_pred).eval(session=get_session()), [1., 0.])
        npt.assert_almost_equal(binary_accuracy(from_logits=False)(y_true, y_pred).eval(session=get_session()),
                                [1. / 2., 0.])

        # =============Percentage Accuracy============= #
        y_pred = tf.Variable([[1., 0., 0.], [1., 0., 0.]])
        y_pred_2 = tf.Variable([[1., 9., 0.], [1., -1., 0.]])
        y_true = tf.Variable([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 1.]])

        npt.assert_array_almost_equal(mean_absolute_percentage_error(y_true, y_pred).eval(session=get_session()),
                                      [50., 50.], decimal=3)
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(mean_absolute_percentage_error(y_true, y_pred).eval(session=get_session()),
                                      mean_absolute_percentage_error(y_true, y_pred_2).eval(session=get_session()),
                                      decimal=3)

        # =============Mean Squared Log Error============= #
        y_pred = tf.Variable([[1., 0., 0.], [1., 0., 0.]])
        y_pred_2 = tf.Variable([[1., 9., 0.], [1., -1., 0.]])
        y_true = tf.Variable([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 1.]])
        npt.assert_array_almost_equal(mean_squared_logarithmic_error(y_true, y_pred).eval(session=get_session()),
                                      [0.24, 0.24], decimal=3)
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(mean_squared_logarithmic_error(y_true, y_pred).eval(session=get_session()),
                                      mean_squared_logarithmic_error(y_true, y_pred_2).eval(session=get_session()),
                                      decimal=3)

    def test_categorical_crossentropy(self):
        y_pred = tf.Variable([[1., 0., 1.], [2., 1., 0.]])
        y_pred_2 = tf.Variable([[1., 2., 1.], [2., 2., 0.]])
        y_true = tf.Variable([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 0.]])
        y_pred_softmax = tf.nn.softmax(y_pred)
        y_pred_2_softmax = tf.nn.softmax(y_pred_2)
        # Truth with Magic number is wrong
        npt.assert_array_almost_equal(categorical_crossentropy(y_true, y_pred_softmax).eval(session=get_session()),
                                      categorical_crossentropy(y_true, y_pred, from_logits=True).eval(
                                          session=get_session()), decimal=3)

    def test_binary_crossentropy(self):
        y_pred = tf.Variable([[0.5, 0., 1.], [2., 0., -1.]])
        y_pred_2 = tf.Variable([[0.5, 2., 1.], [2., 2., -1.]])
        y_true = tf.Variable([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 0.]])
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        y_pred_2_sigmoid = tf.nn.sigmoid(y_pred_2)
        # Truth with Magic number is wrong
        npt.assert_array_almost_equal(binary_crossentropy(y_true, y_pred_sigmoid).eval(session=get_session()),
                                      binary_crosssentropy(y_true, y_pred, from_logits=True).eval(
                                          session=get_session()), decimal=3)
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(
            binary_crossentropy(y_true, y_pred_2, from_logits=True).eval(session=get_session()),
            binary_crossentropy(y_true, y_pred, from_logits=True).eval(session=get_session())
            , decimal=3)
        npt.assert_array_almost_equal(binary_crossentropy(y_true, y_pred_sigmoid).eval(session=get_session()),
                                      binary_crossentropy(y_true, y_pred_2_sigmoid).eval(
                                          session=get_session()), decimal=3)

    def test_negative_log_likelihood(self):
        y_pred = tf.Variable([[0.5, 0., 1.], [2., 0., -1.]])
        y_true = tf.Variable([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 0.]])
        # Truth with Magic number is wrong
        npt.assert_array_almost_equal(nll(y_true, y_pred).eval(session=get_session()), 0.34657377, decimal=3)


if __name__ == '__main__':
    unittest.main()
