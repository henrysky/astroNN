import unittest
import numpy as np

from astroNN.models import Apogee_CNN, Apogee_BCNN, StarNet2017, Apogee_CVAE


class ApogeeModelTestCase(unittest.TestCase):
    def test_apogee_cnn(self):

        # Data preparation, keep the data size large (>800 data points to prevent issues)
        random_xdata = np.random.normal(0, 1, (1000, 7514))
        random_ydata = np.random.normal(0, 1, (1000, 25))

        # Apogee_CNN
        print("======Apogee_CNN======")
        neuralnet = Apogee_CNN()
        print(neuralnet._keras_ver)
        neuralnet.max_epochs = 1
        neuralnet.train(random_xdata, random_ydata)
        prediction = neuralnet.test(random_xdata)
        jacobian = neuralnet.jacobian(random_xdata)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_ydata.shape[1], random_xdata.shape[1], random_xdata.shape[0]])

        # Apogee_BCNN
        print("======Apogee_BCNN======")
        bneuralnet = Apogee_BCNN()
        bneuralnet.max_epochs = 1
        bneuralnet.train(random_xdata, random_ydata)
        prediction, prediction_err = bneuralnet.test(random_xdata)
        jacobian = bneuralnet.jacobian(random_xdata)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_ydata.shape[1], random_xdata.shape[1], random_xdata.shape[0]])

        # StarNet2017
        print("======StarNet2017======")
        starnet2017 = StarNet2017()
        starnet2017.max_epochs = 1
        starnet2017.train(random_xdata, random_ydata)
        prediction = starnet2017.test(random_xdata)
        jacobian = starnet2017.jacobian(random_xdata)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_ydata.shape[1], random_xdata.shape[1], random_xdata.shape[0]])

    def test_apogee_cvae(self):
        # Data preparation, keep the data size large (>800 data points to prevent issues)
        random_xdata = np.random.normal(0, 1, (1000, 7514))

        # Apogee_CVAE
        print("======Apogee_CVAE======")
        cvae_net = Apogee_CVAE()
        cvae_net.max_epochs = 1
        cvae_net.latent_dim = 2
        cvae_net.train(random_xdata, random_xdata)
        prediction = cvae_net.test(random_xdata)
        encoding = cvae_net.test_encoder(random_xdata)

        np.testing.assert_array_equal(prediction.shape, np.expand_dims(random_xdata, axis=-1).shape)


if __name__ == '__main__':
    unittest.main()
