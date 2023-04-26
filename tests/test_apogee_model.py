import os
import requests
import urllib.request
import unittest

import h5py
import numpy as np

from astroNN.models import ApogeeCNN, ApogeeBCNN, ApogeeBCNNCensored, ApogeeDR14GaiaDR2BCNN, StarNet2017, ApogeeCVAE, \
    ApogeeKplerEchelle, ApokascEncoderDecoder
from astroNN.models import load_folder
from astroNN.nn.callbacks import ErrorOnNaN
from astroNN.shared.downloader_tools import TqdmUpTo

from tensorflow import keras as tfk
mnist = tfk.datasets.mnist
utils = tfk.utils

_URL_ORIGIN = "https://www.astro.utoronto.ca/~hleung/shared/ci_data/"
filename = "apogee_dr14_green.h5"
complete_url = _URL_ORIGIN + filename
if not os.path.exists("ci_data"):
    os.mkdir("ci_data")
local_file_path = os.path.join("ci_data", filename)

# Check if files exists
if not os.path.isfile(local_file_path):
    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=complete_url.split("/")[-1]) as t:
        urllib.request.urlretrieve(complete_url, local_file_path, reporthook=t.update_to)
else:
    r = requests.head(complete_url, allow_redirects=True, verify=False)
    assert r.status_code == 200, f"CI data file does not exist on {complete_url}"

# Data preparation
f = h5py.File(local_file_path, "r")
xdata = np.array(f["spectra"])
ydata = np.stack([f["logg"], f["feh"]]).T
ydata_err = np.stack([f["logg_err"], f["feh_err"]]).T


class ApogeeModelTestCase(unittest.TestCase):
    def test_apogee_cnn(self):
        """
        Test ApogeeCNN models
        - training, testing, evaluation
        - basic astroNN model method
        """
        print("======ApogeeCNN======")

        # setup model instance
        neuralnet = ApogeeCNN()
        print(neuralnet)
        # assert no model before training
        self.assertEqual(neuralnet.has_model, False)
        neuralnet.max_epochs = 5  # for quick result
        neuralnet.callbacks = ErrorOnNaN()  # Raise error and fail the test if Nan
        neuralnet.targetname = ["logg", "feh"]
        neuralnet.fit(xdata, ydata)  # training
        neuralnet.fit_on_batch(xdata[:64], ydata[:64])  # single batch fine-tuning test
        # self.assertEqual(neuralnet.uses_learning_phase, True)  # Assert ApogeeCNN uses learning phase (bc of Dropout)

        # test basic astroNN model method
        neuralnet.get_weights()
        neuralnet.summary()
        output_shape = neuralnet.output_shape
        input_shape = neuralnet.input_shape
        neuralnet.get_config()
        neuralnet.save_weights("save_weights_test.h5")  # save astroNN weight only
        neuralnet.plot_dense_stats()
        neuralnet.plot_model()

        prediction = neuralnet.predict(xdata)
        mape = np.median(np.abs(prediction[neuralnet.val_idx] - ydata[neuralnet.val_idx])/ydata[neuralnet.val_idx], axis=0)
        self.assertEqual(np.all(0.15 > mape), True)  # assert less than 15% error
        jacobian = neuralnet.jacobian(xdata[:5])
        # assert shape correct as expected
        np.testing.assert_array_equal(prediction.shape, ydata.shape)
        np.testing.assert_array_equal(jacobian.shape, [xdata[:5].shape[0], ydata.shape[1],
                                                       xdata.shape[1]])

        hessian = neuralnet.hessian(xdata[:5], mean_output=True)
        np.testing.assert_array_equal(hessian.shape, [ydata.shape[1], xdata.shape[1],
                                                      xdata.shape[1]])

        # make sure raised if data dimension not as expected
        self.assertRaises(ValueError, neuralnet.jacobian, np.atleast_3d(xdata[:3]))
        # make sure evaluate run in testing phase instead of learning phase
        # ie no Dropout which makes model deterministic
        self.assertEqual(
            np.all(neuralnet.evaluate(xdata, ydata) == neuralnet.evaluate(xdata, ydata)),
            True)

        # save weight and model again
        neuralnet.save(name="apogee_cnn")
        neuralnet.save_weights("save_weights_test.h5")

        # load the model again
        neuralnet_loaded = load_folder("apogee_cnn")
        neuralnet_loaded.plot_dense_stats()
        # assert has model without training because this is a trained model
        self.assertEqual(neuralnet_loaded.has_model, True)
        # fine tune test
        prediction_loaded = neuralnet_loaded.predict(xdata)

        # ApogeeCNN is deterministic check again
        np.testing.assert_array_equal(prediction, prediction_loaded)

        # Fine tuning test
        neuralnet_loaded.max_epochs = 5
        neuralnet_loaded.callbacks = ErrorOnNaN()
        neuralnet_loaded.fit(xdata, ydata)
        prediction_loaded = neuralnet_loaded.predict(xdata[neuralnet.val_idx])

        # prediction should not be equal after fine-tuning
        self.assertRaises(AssertionError, np.testing.assert_array_equal, prediction, prediction_loaded)

    def test_apogee_bcnn(self):
        """
        Test ApogeeBCNN models
        - training, testing, evaluation
        - Apogee plotting functions
        """

        # ApogeeBCNN
        print("======ApogeeBCNN======")
        bneuralnet = ApogeeBCNN()
        # deliberately chosen targetname to test targetname conversion too
        bneuralnet.targetname = ["logg", "feh"]

        bneuralnet.max_epochs = 5  # for quick result
        bneuralnet.callbacks = ErrorOnNaN()  # Raise error and fail the test if Nan
        bneuralnet.fit(xdata, ydata)
        output_shape = bneuralnet.output_shape
        input_shape = bneuralnet.input_shape

        bneuralnet.mc_num = 2
        prediction, prediction_err = bneuralnet.predict(xdata)
        mape = np.median(np.abs(prediction[bneuralnet.val_idx] - ydata[bneuralnet.val_idx])/ydata[bneuralnet.val_idx], axis=0)
        self.assertEqual(np.all(0.15 > mape), True)  # assert less than 15% error
        self.assertEqual(np.all(0.25 > np.median(prediction_err["total"], axis=0)), True)  # assert entropy
        # assert all of them not equal becaues of MC Dropout
        self.assertEqual(
            np.all(bneuralnet.evaluate(xdata, ydata) != bneuralnet.evaluate(xdata, ydata)),
            True)
        jacobian = bneuralnet.jacobian(xdata[:2], mean_output=True)
        np.testing.assert_array_equal(prediction.shape, ydata.shape)
        bneuralnet.save(name="apogee_bcnn")
        bneuralnet.fit_on_batch(xdata[:64], ydata[:64])  # single batch fine-tuning test

        # just to make sure it can load it back without error
        bneuralnet_loaded = load_folder("apogee_bcnn")

        # prevent memory issue on Tavis CI
        bneuralnet_loaded.mc_num = 2
        pred, pred_err = bneuralnet_loaded.predict(xdata)
        bneuralnet_loaded.save()

        # Fine-tuning test
        bneuralnet_loaded.max_epochs = 5
        bneuralnet_loaded.callbacks = ErrorOnNaN()
        bneuralnet_loaded.fit(xdata, ydata)

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
        bneuralnetcensored.fit(random_xdata, random_ydata)
        # prevent memory issue on Tavis CI
        bneuralnetcensored.mc_num = 2
        prediction, prediction_err = bneuralnetcensored.predict(random_xdata)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        bneuralnetcensored.save(name="apogee_bcnncensored")
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
        self.assertRaises(ValueError, apogeedr14gaiadr2bcnn.train, random_xdata_error2, random_ydata)

        apogeedr14gaiadr2bcnn = ApogeeDR14GaiaDR2BCNN()
        apogeedr14gaiadr2bcnn.max_epochs = 1
        apogeedr14gaiadr2bcnn.callbacks = ErrorOnNaN()
        apogeedr14gaiadr2bcnn.fit(random_xdata, random_ydata)

        # prevent memory issue on Tavis CI
        apogeedr14gaiadr2bcnn.mc_num = 2
        prediction, prediction_err = apogeedr14gaiadr2bcnn.predict(random_xdata)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        apogeedr14gaiadr2bcnn.save(name="apogeedr14_gaiadr2")
        bneuralnetcensored_loaded = load_folder("apogeedr14_gaiadr2")

    def test_apogee_cvae(self):
        # ApogeeCVAE
        print("======ApogeeCVAE======")
        cvae_net = ApogeeCVAE()
        cvae_net.max_epochs = 3
        cvae_net.latent_dim = 2
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net.fit(xdata, xdata)
        prediction = cvae_net.predict(xdata)
        cvae_net.fit_on_batch(xdata, xdata)
        _, _, encoding = cvae_net.predict_encoder(xdata)
        print(cvae_net.evaluate(xdata, xdata))

        np.testing.assert_array_equal(prediction.shape, np.expand_dims(xdata, axis=-1).shape)
        np.testing.assert_array_equal(encoding.shape, [xdata.shape[0], cvae_net.latent_dim])
        cvae_net.save(name="apogee_cvae")

        # just to make sure it can load it back without error
        cvae_net_loaded = load_folder("apogee_cvae")
        _, encoding_stdev, encoding = cvae_net_loaded.predict_encoder(xdata)
        np.testing.assert_array_equal(encoding.shape, [xdata.shape[0], cvae_net.latent_dim])
        assert np.all(encoding_stdev > 0)

        # Fine-tuning test
        cvae_net_loaded.max_epochs = 1
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net_loaded.fit(xdata, xdata)

    def test_starnet2017(self):
        """
        Test StarNet2017 models
        - training, testing
        """
        # Data preparation, keep the data size large (>800 data points to prevent issues)
        random_xdata = np.random.normal(0, 1, (200, 7514))
        random_ydata = np.random.normal(0, 1, (200, 25))

        # StarNet2017
        print("======StarNet2017======")
        starnet2017 = StarNet2017()
        starnet2017.max_epochs = 1
        starnet2017.callbacks = ErrorOnNaN()
        starnet2017.fit(random_xdata, random_ydata)
        prediction = starnet2017.predict(random_xdata)
        np.testing.assert_array_equal(prediction.shape, random_ydata.shape)
        starnet2017.save(name="starnet2017")

    def test_ApogeeKplerEchelle(self):
        """
        Test ApogeeKplerEchelle models
        - training, testing
        """
        # Data preparation, keep the data size large (>800 data points to prevent issues)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = utils.to_categorical(y_train, 10)
        y_test = utils.to_categorical(y_test, 10)
        # To convert to desirable type
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        print("======ApogeeKplerEchelle======")
        apokasc_nn = ApogeeKplerEchelle()
        apokasc_nn.max_epochs = 2
        apokasc_nn.dropout_rate = 0.
        apokasc_nn.input_norm_mode = {"input": 255, "aux": 0}
        apokasc_nn.labels_norm_mode = 0
        apokasc_nn.task = "classification"
        apokasc_nn.callbacks = ErrorOnNaN()
        apokasc_nn.fit({"input": x_train, "aux": y_train}, {"output": y_train})
        prediction = apokasc_nn.predict({"input": x_train, "aux": y_train})
        # we ave the answer as aux input so the prediction should be near perfect
        total_num = y_train.shape[0]
        assert np.sum((prediction>0.5) == (y_train>0.5)) > total_num * 0.99
        apokasc_nn.save(name="apokasc_nn")

        apokasc_nn_reloaded = load_folder("apokasc_nn")
        prediction_reloaded = apokasc_nn_reloaded.predict({"input": x_train, "aux": y_train})
        np.testing.assert_array_equal(prediction, prediction_reloaded)
        
        
    def test_apogee_identical_transfer(self):
        """
        Test transfer learning function on two identical model to make sure 100% weights get transferred
        """
        from astroNN.nn.metrics import median_absolute_deviation

        # ApogeeCNN
        print("======ApogeeCNN Transfer Learning Identical======")
        neuralnet = ApogeeCNN()
        # deliberately chosen targetname to test targetname conversion too
        neuralnet.targetname = ["logg", "feh"]

        neuralnet.max_epochs = 20
        neuralnet.callbacks = ErrorOnNaN()  # Raise error and fail the test if Nan
        neuralnet.fit(xdata[:5000, :1500], ydata[:5000])
        pred = neuralnet.predict(xdata[:5000, :1500][neuralnet.val_idx])
        
        neuralnet2 = ApogeeCNN()
        neuralnet2.max_epochs = 1
        # initialize with the correct shape
        neuralnet2.fit(xdata[:5000, :1500], ydata[:5000])
        # transfer weight
        neuralnet2.transfer_weights(neuralnet)
        pred2 = neuralnet2.predict(xdata[:5000, :1500][neuralnet.val_idx])
        mad_1 = median_absolute_deviation(pred[:, 0], ydata[neuralnet.val_idx][:, 0], axis=None).numpy()
        mad_2 = median_absolute_deviation(pred2[:, 0], ydata[neuralnet.val_idx][:, 0], axis=None).numpy()
        
        # accurancy sould be very similar as they are the same model
        self.assertAlmostEqual(mad_1, mad_2)
        

    def test_apogee_transferlearning(self):
        """
        Test transfer learning function
        """
        from astroNN.nn.metrics import median_absolute_deviation

        # ApogeeBCNN
        print("======ApogeeBCNN Transfer Learning======")
        bneuralnet = ApogeeBCNN()
        # deliberately chosen targetname to test targetname conversion too
        bneuralnet.targetname = ["logg", "feh"]

        bneuralnet.max_epochs = 20
        bneuralnet.callbacks = ErrorOnNaN()  # Raise error and fail the test if Nan
        bneuralnet.fit(xdata[:5000, :1500], ydata[:5000])
        pred = bneuralnet.predict(xdata[:5000, :1500][bneuralnet.val_idx])
        
        bneuralnet2 = ApogeeBCNN()
        bneuralnet2.max_epochs = 1
        # initialize with the correct shape
        bneuralnet2.fit(xdata[5000:, 1500:], ydata[5000:])
        # transfer weight
        bneuralnet2.transfer_weights(bneuralnet)
        bneuralnet2.max_epochs = 10
        bneuralnet2.fit(xdata[5000:, 1500:], ydata[5000:])
        pred2 = bneuralnet2.predict(xdata[5000:, 1500:][bneuralnet2.val_idx])
        median_absolute_deviation(pred[0][:, 0], ydata[bneuralnet.val_idx][:, 0], axis=None).numpy()
        median_absolute_deviation(pred2[0][:, 0], ydata[5000:, 0][bneuralnet2.val_idx], axis=None).numpy()
        
        # transferred weights should be untrainable thus stay the same
        np.testing.assert_array_equal(bneuralnet.keras_model.weights[6], bneuralnet2.keras_model.weights[6])

    def test_apokaasc_encoder_decoder(self):
        
        nn_output_internal = xdata.shape[1] // 4
        xdata_vae = xdata[:, :nn_output_internal*4]
        
        # ApokascEncoderDecoder
        print("======ApokascEncoderDecoder======")
        cvae_net = ApokascEncoderDecoder()
        cvae_net.num_filters =  [8, 16, 8, 8]
        cvae_net.max_epochs = 3
        cvae_net.latent_dim = 2
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net.fit(xdata_vae, xdata_vae)
        prediction = cvae_net.predict(xdata_vae)
        cvae_net.fit_on_batch(xdata_vae, xdata_vae)
        _, _, encoding = cvae_net.predict_encoder(xdata_vae)
        print(cvae_net.evaluate(xdata_vae, xdata_vae))

        np.testing.assert_array_equal(prediction.shape, np.expand_dims(xdata_vae, axis=-1).shape)
        np.testing.assert_array_equal(encoding.shape, [xdata_vae.shape[0], cvae_net.latent_dim])
        cvae_net.save(name="apokasc_encoderdecoder")

        # just to make sure it can load it back without error
        cvae_net_loaded = load_folder("apokasc_encoderdecoder")
        _, encoding_stdev, encoding = cvae_net_loaded.predict_encoder(xdata_vae)
        np.testing.assert_array_equal(encoding.shape, [xdata_vae.shape[0], cvae_net.latent_dim])
        assert np.all(encoding_stdev > 0)

        # Fine-tuning test
        cvae_net_loaded.max_epochs = 1
        cvae_net.callbacks = ErrorOnNaN()
        cvae_net_loaded.fit(xdata_vae, xdata_vae)


if __name__ == "__main__":
    unittest.main()
