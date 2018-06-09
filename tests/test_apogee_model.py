import unittest

import numpy as np

from astroNN.models import ApogeeCNN, ApogeeBCNN, ApogeeBCNNCensored, StarNet2017, ApogeeCVAE
from astroNN.models import load_folder
from astroNN.nn.callbacks import ErrorOnNaN

# Data preparation, keep the data size large (>800 data points to prevent issues)
random_xdata = np.random.normal(0, 1, (200, 7514))
random_ydata = np.random.normal(0, 1, (200, 25))


class ApogeeModelTestCase(unittest.TestCase):
    def test_apogee_cnn(self):
        # ApogeeCNN
        print("======ApogeeCNN======")
        neuralnet = ApogeeCNN()
        self.assertEqual(neuralnet.has_model, False)
        neuralnet.max_epochs = 1
        neuralnet.callbacks = ErrorOnNaN()
        neuralnet.train(random_xdata, random_ydata)
        self.assertEqual(neuralnet.uses_learning_phase, True)
        neuralnet.get_weights()
        neuralnet.get_config()
        neuralnet.save_weights('save_weights_test.h5')
        neuralnet.summary()
        output_shape = neuralnet.output_shape
        input_shape = neuralnet.input_shape
        prediction = neuralnet.test(random_xdata)
        jacobian = neuralnet.jacobian(random_xdata[:10])
        self.assertRaises(ValueError, neuralnet.jacobian, np.atleast_3d(random_xdata[:10]))
        # make sure evaluate run in testing phase instead of learning phase
        # ie no Dropout which makes model deterministic
        self.assertEqual(
            np.all(neuralnet.evaluate(random_xdata, random_ydata) == neuralnet.evaluate(random_xdata, random_ydata)),
            True)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_xdata[:10].shape[0], random_ydata.shape[1],
                                                       random_xdata.shape[1]])
        neuralnet.save(name='apogee_cnn')
        neuralnet.save_weights('save_weights_test.h5')

        neuralnet_loaded = load_folder("apogee_cnn")
        self.assertEqual(neuralnet_loaded.has_model, True)
        neuralnet_loaded.max_epochs = 1
        neuralnet_loaded.callbacks = ErrorOnNaN()
        prediction_loaded = neuralnet_loaded.test(random_xdata)

        # Apogee_CNN is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)

        # Fine tuning test
        neuralnet_loaded.train(random_xdata, random_ydata)
        prediction_loaded = neuralnet_loaded.test(random_xdata)
        # prediction should not be equal after fine-tuning
        self.assertRaises(AssertionError, np.testing.assert_array_equal, prediction, prediction_loaded)

    def test_apogee_bcnn(self):
        random_xdata = np.random.normal(0, 1, (200, 7514))
        random_ydata = np.random.normal(0, 1, (200, 7))

        # ApogeeBCNN
        print("======ApogeeBCNN======")
        bneuralnet = ApogeeBCNN()
        bneuralnet.targetname = ['teff', 'logg', 'M', 'alpha', 'C1', 'Ti', 'Ti2']

        bneuralnet.max_epochs = 1
        bneuralnet.callbacks = ErrorOnNaN()
        bneuralnet.train(random_xdata, random_ydata)
        output_shape = bneuralnet.output_shape
        input_shape = bneuralnet.input_shape
        # prevent memory issue on Tavis CI
        bneuralnet.mc_num = 3
        prediction, prediction_err = bneuralnet.test(random_xdata)

        print(bneuralnet.evaluate(random_xdata, random_ydata))

        bneuralnet.plot_dense_stats()
        bneuralnet.plot_model()
        jacobian = bneuralnet.jacobian(random_xdata[:10], mean_output=True)

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        bneuralnet.save(name='apogee_bcnn')

        # just to make sure it can load it back without error
        bneuralnet_loaded = load_folder("apogee_bcnn")
        bneuralnet_loaded.plot_dense_stats()
        bneuralnet_loaded.callbacks = ErrorOnNaN()

        # prevent memory issue on Tavis CI
        bneuralnet_loaded.mc_num = 3
        pred, pred_err = bneuralnet_loaded.test(random_xdata)
        bneuralnet_loaded.aspcap_residue_plot(pred, pred, pred_err['total'])
        bneuralnet_loaded.jacobian_aspcap(jacobian)
        bneuralnet_loaded.save()

        # Fine-tuning test
        bneuralnet_loaded.max_epochs = 1
        bneuralnet_loaded.train(random_xdata, random_ydata)

        pred, pred_err = bneuralnet_loaded.test_old(random_xdata)

    def test_apogee_bcnnconsered(self):
        random_xdata = np.random.normal(0, 1, (200, 7514))

        # ApogeeBCNNCensored
        print("======ApogeeBCNNCensored======")
        bneuralnetcensored = ApogeeBCNNCensored()
        datalen = len(bneuralnetcensored.targetname)
        random_ydata = np.random.normal(0, 1, (200, datalen))

        bneuralnetcensored.max_epochs = 1
        bneuralnetcensored.callbacks = ErrorOnNaN()
        bneuralnetcensored.train(random_xdata, random_ydata)
        output_shape = bneuralnetcensored.output_shape
        input_shape = bneuralnetcensored.input_shape
        # prevent memory issue on Tavis CI
        bneuralnetcensored.mc_num = 3
        prediction, prediction_err = bneuralnetcensored.test(random_xdata)

        print(bneuralnetcensored.evaluate(random_xdata, random_ydata))

        jacobian = bneuralnetcensored.jacobian(random_xdata[:10], mean_output=True)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        bneuralnetcensored.save(name='apogee_bcnncensored')

        bneuralnetcensored_loaded = load_folder("apogee_bcnncensored")

    def test_apogee_cvae(self):
        # Data preparation, keep the data size large (>800 data points to prevent issues)
        random_xdata = np.random.normal(0, 1, (1000, 7514))

        # ApogeeCVAE
        print("======ApogeeCVAE======")
        cvae_net = ApogeeCVAE()
        cvae_net.max_epochs = 1
        cvae_net.latent_dim = 2
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net.train(random_xdata, random_xdata)
        prediction = cvae_net.test(random_xdata)
        encoding = cvae_net.test_encoder(random_xdata)
        cvae_net.evaluate(random_xdata, random_xdata)

        np.testing.assert_array_equal(prediction.shape, np.expand_dims(random_xdata, axis=-1).shape)
        np.testing.assert_array_equal(encoding.shape, [random_xdata.shape[0], cvae_net.latent_dim])
        cvae_net.save(name='apogee_cvae')

        # just to make sure it can load it back without error
        cvae_net_loaded = load_folder("apogee_cvae")
        encoding = cvae_net_loaded.test_encoder(random_xdata)
        np.testing.assert_array_equal(encoding.shape, [random_xdata.shape[0], cvae_net.latent_dim])

        # Fine-tuning test
        cvae_net_loaded.max_epochs = 1
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net_loaded.train(random_xdata, random_xdata)

    def test_starnet2017(self):
        # StarNet2017
        print("======StarNet2017======")
        starnet2017 = StarNet2017()
        starnet2017.max_epochs = 1
        starnet2017.callbacks = ErrorOnNaN()
        starnet2017.train(random_xdata, random_ydata)
        prediction = starnet2017.test(random_xdata)
        jacobian = starnet2017.jacobian(random_xdata[:10])

        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_xdata[:10].shape[0], random_ydata.shape[1],
                                                       random_xdata.shape[1]])
        starnet2017.save(name='starnet2017')

        starnet2017_loaded = load_folder("starnet2017")
        prediction_loaded = starnet2017_loaded.test(random_xdata)
        # StarNet2017 is deterministic
        np.testing.assert_array_equal(prediction, prediction_loaded)


if __name__ == '__main__':
    unittest.main()
