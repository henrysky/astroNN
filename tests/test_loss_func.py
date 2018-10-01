import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from astroNN.config import MAGIC_NUMBER, keras_import_manager
from astroNN.nn import magic_correction_term, reduce_var
from astroNN.nn.losses import mean_absolute_error, mean_squared_error, categorical_crossentropy, binary_crossentropy, \
    nll, mean_error, zeros_loss, mean_percentage_error
from astroNN.nn.metrics import categorical_accuracy, binary_accuracy, mean_absolute_percentage_error, \
    mean_squared_logarithmic_error

keras = keras_import_manager()
get_session = keras.backend.get_session

# force these tests to use CPU, using GPU will be much slower for such small tests
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
keras.backend.set_session(sess)


class LossFuncTestCase(unittest.TestCase):
    def test_loss_func_util(self):
        # make sure custom reduce_var works
        content = [1, 2, 3, 4, 5]
        var_array = tf.constant(content)
        
        self.assertEqual(reduce_var(var_array).eval(session=get_session()), np.var(content))

    def test_loss_magic(self):
        # =============Magic correction term============= #
        y_true = tf.constant([[2., MAGIC_NUMBER, MAGIC_NUMBER], [2., MAGIC_NUMBER, 4.]])

        npt.assert_array_equal(magic_correction_term(y_true).eval(session=get_session()), [3., 1.5])

    def test_loss_mse(self):
        # =============MSE/MAE============= #
        y_pred = tf.constant([[2., 3., 4.], [2., 3., 7.]])
        y_pred_2 = tf.constant([[2., 9., 4.], [2., 0., 7.]])
        y_true = tf.constant([[2., MAGIC_NUMBER, 4.], [2., MAGIC_NUMBER, 4.]])
        
        npt.assert_almost_equal(mean_absolute_error(y_true, y_pred).eval(session=get_session()), [0., 3. / 2.])
        npt.assert_almost_equal(mean_squared_error(y_true, y_pred).eval(session=get_session()), [0., 9. / 2])

        # make sure neural network prediction won't matter for magic number term
        npt.assert_almost_equal(mean_absolute_error(y_true, y_pred).eval(session=get_session()),
                                mean_absolute_error(y_true, y_pred_2).eval(session=get_session()))
        npt.assert_almost_equal(mean_squared_error(y_true, y_pred).eval(session=get_session()),
                                mean_squared_error(y_true, y_pred_2).eval(session=get_session()))

    def test_loss_mean_err(self):
        # =============Mean Error============= #
        y_pred = tf.constant([[1., 3., 4.], [2., 3., 7.]])
        y_true = tf.constant([[2., MAGIC_NUMBER, 3.], [2., MAGIC_NUMBER, 7.]])
        
        npt.assert_almost_equal(mean_error(y_true, y_pred).eval(session=get_session()), [0., 0.])

    def test_loss_acurrancy(self):
        # =============Accuracy============= #
        y_pred = tf.constant([[1., 0., 0.], [1., 0., 0.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [0., MAGIC_NUMBER, 1.]])
        
        npt.assert_array_equal(categorical_accuracy(y_true, y_pred).eval(session=get_session()), [1., 0.])
        npt.assert_almost_equal(binary_accuracy(from_logits=False)(y_true, y_pred).eval(session=get_session()),
                                [1. / 2., 0.])

    def test_loss_abs_error(self):
        # =============Abs Percentage Accuracy============= #
        y_pred = tf.constant([[1., 0., 0.], [1., 0., 0.]])
        y_pred_2 = tf.constant([[1., 9., 0.], [1., -1., 0.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 1.]])
        

        npt.assert_array_almost_equal(mean_absolute_percentage_error(y_true, y_pred).eval(session=get_session()),
                                      [50., 50.], decimal=3)
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(mean_absolute_percentage_error(y_true, y_pred).eval(session=get_session()),
                                      mean_absolute_percentage_error(y_true, y_pred_2).eval(session=get_session()),
                                      decimal=3)

    def test_loss_percentage_error(self):
        # =============Percentage Accuracy============= #
        y_pred = tf.constant([[1., 0., 0.], [1., 0., 0.]])
        y_pred_2 = tf.constant([[1., 9., 0.], [1., -1., 0.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 1.]])
        

        npt.assert_array_almost_equal(mean_percentage_error(y_true, y_pred).eval(session=get_session()),
                                      [50., 50.], decimal=3)
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(mean_percentage_error(y_true, y_pred).eval(session=get_session()),
                                      mean_percentage_error(y_true, y_pred_2).eval(session=get_session()),
                                      decimal=3)

    def test_loss_log_error(self):
        # =============Mean Squared Log Error============= #
        y_pred = tf.constant([[1., 0., 0.], [1., 0., 0.]])
        y_pred_2 = tf.constant([[1., 9., 0.], [1., -1., 0.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 1.]])
        
        npt.assert_array_almost_equal(mean_squared_logarithmic_error(y_true, y_pred).eval(session=get_session()),
                                      [0.24, 0.24], decimal=3)
        # make sure neural network prediction won't matter for magic number term
        npt.assert_array_almost_equal(mean_squared_logarithmic_error(y_true, y_pred).eval(session=get_session()),
                                      mean_squared_logarithmic_error(y_true, y_pred_2).eval(session=get_session()),
                                      decimal=3)

    def test_loss_zeros(self):
        # =============Zeros Loss============= #
        y_pred = tf.constant([[1., 0., 0.], [5., -9., 2.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 1.]])
        
        npt.assert_array_almost_equal(zeros_loss(y_true, y_pred).eval(session=get_session()), [0., 0.])

    def test_categorical_crossentropy(self):
        y_pred = tf.constant([[1., 0., 1.], [2., 1., 0.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 0.]])
        
        y_pred_softmax = tf.nn.softmax(y_pred)
        # Truth with Magic number is wrong
        npt.assert_array_almost_equal(categorical_crossentropy(y_true, y_pred_softmax).eval(session=get_session()),
                                      categorical_crossentropy(y_true, y_pred, from_logits=True).eval(
                                          session=get_session()), decimal=3)

    def test_binary_crossentropy(self):
        y_pred = tf.constant([[0.5, 0., 1.], [2., 0., -1.]])
        y_pred_2 = tf.constant([[0.5, 2., 1.], [2., 2., -1.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 0.]])
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        y_pred_2_sigmoid = tf.nn.sigmoid(y_pred_2)
        # Truth with Magic number is wrong
        
        npt.assert_array_almost_equal(binary_crossentropy(y_true, y_pred_sigmoid).eval(session=get_session()),
                                      binary_crossentropy(y_true, y_pred, from_logits=True).eval(
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
        y_pred = tf.constant([[0.5, 0., 1.], [2., 0., -1.]])
        y_true = tf.constant([[1., MAGIC_NUMBER, 1.], [1., MAGIC_NUMBER, 0.]])
        
        npt.assert_array_almost_equal(nll(y_true, y_pred).eval(session=get_session()), 0.34657377, decimal=3)


if __name__ == '__main__':
    unittest.main()
