import unittest
import numpy as np

from astroNN.models import Apogee_CNN, Apogee_BCNN, StarNet2017


class ApogeeModelTestCase(unittest.TestCase):
    def test_apogee_cnn(self):

        # Data preparation
        random_xdata = np.random.random((1000,7514))
        random_ydata = np.random.random((1000, 25))

        # Apogee_CNN
        neuralnet = Apogee_CNN()
        neuralnet.max_epochs = 1
        neuralnet.train(random_xdata, random_ydata)
        prediction = neuralnet.test(random_xdata)
        jacobian = neuralnet.jacobian(random_xdata)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_ydata.shape[1], random_xdata.shape[1], random_xdata.shape[0]])

        # Apogee_BCNN
        bneuralnet = Apogee_BCNN()
        bneuralnet.max_epochs = 1
        bneuralnet.train(random_xdata, random_ydata)
        prediction, prediction_err, model_err, predictive_err = bneuralnet.test(random_xdata)
        jacobian = neuralnet.jacobian(random_xdata)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_ydata.shape[1], random_xdata.shape[1], random_xdata.shape[0]])

        # StarNet2017
        starnet2017 = StarNet2017()
        starnet2017.max_epochs = 1
        starnet2017.train(random_xdata, random_ydata)
        prediction = starnet2017.test(random_xdata)
        jacobian = neuralnet.jacobian(random_xdata)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_ydata.shape[1], random_xdata.shape[1], random_xdata.shape[0]])

if __name__ == '__main__':
    unittest.main()