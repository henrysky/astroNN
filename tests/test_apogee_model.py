import unittest

import numpy as np
from astroNN.models import ApogeeCNN, ApogeeBCNN, ApogeeBCNNCensored, ApogeeDR14GaiaDR2BCNN, StarNet2017, ApogeeCVAE
from astroNN.models import load_folder
from astroNN.nn.callbacks import ErrorOnNaN


class ApogeeModelTestCase(unittest.TestCase):
    def test_apogee_cnn(self):
        """
        Test ApogeeCNN models
        - training, testing, evaluation
        - basic astroNN model method
        """
        print("======ApogeeCNN======")

        # Data preparation
        random_xdata = np.random.normal(0, 1, (200, 1024))
        random_ydata = np.random.normal(0, 1, (200, 2))

        # setup model instance
        neuralnet = ApogeeCNN()
        print(neuralnet)
        # assert no model before training
        self.assertEqual(neuralnet.has_model, False)
        neuralnet.max_epochs = 1  # for quick result
        neuralnet.callbacks = ErrorOnNaN()  # Raise error and fail the test if Nan
        neuralnet.train(random_xdata, random_ydata)  # training
        neuralnet.train_on_batch(random_xdata, random_ydata)  # single batch fine-tuning test
        self.assertEqual(neuralnet.uses_learning_phase, True)  # Assert ApogeeCNN uses learning phase (bc of Dropout)

        # test basic astroNN model method
        neuralnet.get_weights()
        neuralnet.summary()
        output_shape = neuralnet.output_shape
        input_shape = neuralnet.input_shape
        neuralnet.get_config()
        neuralnet.save_weights('save_weights_test.h5')  # save astroNN weight only
        neuralnet.plot_dense_stats()
        neuralnet.plot_model()

        prediction = neuralnet.test(random_xdata)
        jacobian = neuralnet.jacobian(random_xdata[:2])
        hessian = neuralnet.hessian_diag(random_xdata[:2])
        hessian_full_approx = neuralnet.hessian(random_xdata[:2], method='approx')
        hessian_full_exact = neuralnet.hessian(random_xdata[:2], method='exact')

        #  make sure raised if data dimension not as expected
        self.assertRaises(ValueError, neuralnet.jacobian, np.atleast_3d(random_xdata[:3]))
        # make sure evaluate run in testing phase instead of learning phase
        # ie no Dropout which makes model deterministic
        self.assertEqual(
            np.all(neuralnet.evaluate(random_xdata, random_ydata) == neuralnet.evaluate(random_xdata, random_ydata)),
            True)

        # assert shape correct as expected
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [random_xdata[:2].shape[0], random_ydata.shape[1],
                                                       random_xdata.shape[1]])
        np.testing.assert_array_equal(hessian.shape, [random_xdata[:2].shape[0], random_ydata.shape[1],
                                                      random_xdata.shape[1]])
        # hessian approx and exact result should have the same shape
        np.testing.assert_array_equal(hessian_full_approx.shape, hessian_full_exact.shape)

        # save weight and model again
        neuralnet.save(name='apogee_cnn')
        neuralnet.save_weights('save_weights_test.h5')

        # load the model again
        neuralnet_loaded = load_folder("apogee_cnn")
        neuralnet_loaded.plot_dense_stats()
        # assert has model without training because this is a trained model
        self.assertEqual(neuralnet_loaded.has_model, True)
        # fine tune test
        prediction_loaded = neuralnet_loaded.test(random_xdata)

        # ApogeeCNN is deterministic check again
        np.testing.assert_array_equal(prediction, prediction_loaded)

        # Fine tuning test
        neuralnet_loaded.max_epochs = 1
        neuralnet_loaded.callbacks = ErrorOnNaN()
        neuralnet_loaded.train(random_xdata, random_ydata)
        prediction_loaded = neuralnet_loaded.test(random_xdata)

        # prediction should not be equal after fine-tuning
        self.assertRaises(AssertionError, np.testing.assert_array_equal, prediction, prediction_loaded)

    def test_apogee_bcnn(self):
        """
        Test ApogeeBCNN models
        - training, testing, evaluation
        - Apogee plotting functions
        """
        # Data preparation
        random_xdata = np.random.normal(0, 1, (200, 7514))
        random_ydata = np.random.normal(0, 1, (200, 7))

        # ApogeeBCNN
        print("======ApogeeBCNN======")
        bneuralnet = ApogeeBCNN()
        # deliberately chosen targetname to test targetname conversion too
        bneuralnet.targetname = ['teff', 'logg', 'M', 'alpha', 'C1', 'Ti', 'Ti2']

        bneuralnet.max_epochs = 1  # for quick result
        bneuralnet.callbacks = ErrorOnNaN()  # Raise error and fail the test if Nan
        bneuralnet.train(random_xdata, random_ydata)
        output_shape = bneuralnet.output_shape
        input_shape = bneuralnet.input_shape
        # prevent memory issue on Tavis CI so set mc_num=2
        bneuralnet.mc_num = 2
        prediction, prediction_err = bneuralnet.test(random_xdata)
        # assert all of them not equal becaues of MC Dropout
        self.assertEqual(
            np.all(bneuralnet.evaluate(random_xdata, random_ydata) != bneuralnet.evaluate(random_xdata, random_ydata)),
            True)
        jacobian = bneuralnet.jacobian(random_xdata[:2], mean_output=True)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        bneuralnet.save(name='apogee_bcnn')
        bneuralnet.train_on_batch(random_xdata, random_ydata)  # single batch fine-tuning test

        # just to make sure it can load it back without error
        bneuralnet_loaded = load_folder("apogee_bcnn")

        # prevent memory issue on Tavis CI
        bneuralnet_loaded.mc_num = 2
        pred, pred_err = bneuralnet_loaded.test(random_xdata)
        bneuralnet_loaded.aspcap_residue_plot(pred, pred, pred_err['total'])
        bneuralnet_loaded.jacobian_aspcap(jacobian)
        bneuralnet_loaded.save()

        # Fine-tuning test
        bneuralnet_loaded.max_epochs = 1
        bneuralnet_loaded.callbacks = ErrorOnNaN()
        bneuralnet_loaded.train(random_xdata, random_ydata)
        pred, pred_err = bneuralnet_loaded.test_old(random_xdata)

    def test_apogee_bcnnconsered(self):
        """
        Test ApogeeBCNNCensored models
        - training, testing, evaluation
        """
        # Data preparation
        random_xdata = np.random.normal(0, 1, (200, 7514))

        # ApogeeBCNNCensored
        print("======ApogeeBCNNCensored======")
        bneuralnetcensored = ApogeeBCNNCensored()
        datalen = len(bneuralnetcensored.targetname)
        random_ydata = np.random.normal(0, 1, (200, datalen))

        bneuralnetcensored.max_epochs = 1
        bneuralnetcensored.callbacks = ErrorOnNaN()
        bneuralnetcensored.train(random_xdata, random_ydata)
        # prevent memory issue on Tavis CI
        bneuralnetcensored.mc_num = 2
        prediction, prediction_err = bneuralnetcensored.test(random_xdata)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        bneuralnetcensored.save(name='apogee_bcnncensored')
        bneuralnetcensored_loaded = load_folder("apogee_bcnncensored")

    def test_apogeedr14_gaiadr2(self):
        """
        Test ApogeeDR14GaiaDR2BCNN models
        - training, testing, evaluation
        """
        # Data preparation
        random_xdata_error1 = np.random.normal(0, 1, (200, 7514))
        random_xdata_error2 = np.random.normal(0, 1, (200, 7515))
        random_xdata = np.random.normal(0, 1, (200, 7516))

        # ApogeeBCNNCensored
        print("======ApogeeDR14GaiaDR2BCNN======")
        random_ydata = np.random.normal(0, 1, (200, 1))

        apogeedr14gaiadr2bcnn = ApogeeDR14GaiaDR2BCNN()
        apogeedr14gaiadr2bcnn.max_epochs = 1
        apogeedr14gaiadr2bcnn.callbacks = ErrorOnNaN()
        self.assertRaises(IndexError, apogeedr14gaiadr2bcnn.train, random_xdata_error1, random_ydata)

        apogeedr14gaiadr2bcnn = ApogeeDR14GaiaDR2BCNN()
        apogeedr14gaiadr2bcnn.max_epochs = 1
        apogeedr14gaiadr2bcnn.callbacks = ErrorOnNaN()
        apogeedr14gaiadr2bcnn.train(random_xdata_error2, random_ydata)
        self.assertRaises(ValueError, apogeedr14gaiadr2bcnn.train, random_xdata_error2, random_ydata)

        apogeedr14gaiadr2bcnn = ApogeeDR14GaiaDR2BCNN()
        apogeedr14gaiadr2bcnn.max_epochs = 1
        apogeedr14gaiadr2bcnn.callbacks = ErrorOnNaN()
        apogeedr14gaiadr2bcnn.train(random_xdata, random_ydata)

        # prevent memory issue on Tavis CI
        apogeedr14gaiadr2bcnn.mc_num = 2
        prediction, prediction_err = apogeedr14gaiadr2bcnn.test(random_xdata)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        apogeedr14gaiadr2bcnn.save(name='apogeedr14_gaiadr2')
        bneuralnetcensored_loaded = load_folder("apogeedr14_gaiadr2")

    def test_apogee_cvae(self):
        # Data preparation
        random_xdata = np.random.normal(0, 1, (200, 7514))

        # ApogeeCVAE
        print("======ApogeeCVAE======")
        cvae_net = ApogeeCVAE()
        cvae_net.max_epochs = 1
        cvae_net.latent_dim = 2
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net.train(random_xdata, random_xdata)
        prediction = cvae_net.test(random_xdata)
        cvae_net.train_on_batch(random_xdata, random_xdata)
        encoding = cvae_net.test_encoder(random_xdata)
        print(cvae_net.evaluate(random_xdata, random_xdata))

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
        """
        Test StarNet2017 models
        - training, testing, evaluation
        """
        # Data preparation, keep the data size large (>800 data points to prevent issues)
        random_xdata = np.random.normal(0, 1, (200, 7514))
        random_ydata = np.random.normal(0, 1, (200, 25))

        # StarNet2017
        print("======StarNet2017======")
        starnet2017 = StarNet2017()
        starnet2017.max_epochs = 1
        starnet2017.callbacks = ErrorOnNaN()
        starnet2017.train(random_xdata, random_ydata)
        prediction = starnet2017.test(random_xdata)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        starnet2017.save(name='starnet2017')


if __name__ == '__main__':
    unittest.main()
