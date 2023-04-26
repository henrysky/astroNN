##################################################################
##
## To make sure publish models in papers actually still work fine
##
##################################################################

import os
import unittest
import subprocess

import numpy as np
from astroNN.models import load_folder

ci_data_folder = "ci_data"
if not os.path.exists(ci_data_folder):
    os.mkdir(ci_data_folder)


def download_models(models_url):
    """
    function to download model directly from github url
    """
    for model_url in models_url:
        model_folder_name = os.path.basename(model_url)
        if not os.path.exists(os.path.join(ci_data_folder, model_folder_name)):
            download_args = ["svn", "export", model_url, os.path.join(ci_data_folder, model_folder_name)]
            res = subprocess.Popen(download_args, stdout=subprocess.PIPE)
            output, _error = res.communicate()
            if not _error:
                pass
            else:
                raise ConnectionError(f"Error downloading the models {model_url}")
        else:  # if the model is cached on Github Action, do a sanity check on remote folder without downloading it
            check_args = ["svn", "log", model_url]
            result = subprocess.Popen(check_args)
            text = result.communicate()[0]
            assert result.returncode == 0, f"Remote folder does not exist at {model_url}"


class PapersModelsCase(unittest.TestCase):
    """
    Make sure all paper model work correctly
    """

    def test_arXiv_1808_04428(self):
        """
        original astroNN paper models for deep learning of multi-elemental abundances
        """
        from astroNN.apogee import visit_spectra, apogee_continuum
        from astropy.io import fits

        # first model
        models_url = [
            "https://github.com/henrysky/astroNN_spectra_paper_figures/trunk/astroNN_0606_run001",
            "https://github.com/henrysky/astroNN_spectra_paper_figures/trunk/astroNN_0617_run001",
        ]
        download_models(models_url)

        opened_fits = fits.open(
            visit_spectra(dr=14, location=4405, apogee="2M19060637+4717296")
        )
        spectrum = opened_fits[1].data
        spectrum_err = opened_fits[2].data
        spectrum_bitmask = opened_fits[3].data

        # using default continuum and bitmask values to continuum normalize
        norm_spec, norm_spec_err = apogee_continuum(
            spectrum, spectrum_err, bitmask=spectrum_bitmask, dr=14
        )
        # load neural net
        neuralnet = load_folder(os.path.join(ci_data_folder, "astroNN_0617_run001"))

        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)

        # assert temperature and gravity okay
        self.assertTrue(np.all(pred[0, 0:2] > [4700.0, 2.40]))
        self.assertTrue(np.all(pred[0, 0:2] < [4750.0, 2.47]))

        # load neural net
        neuralnet = load_folder(os.path.join(ci_data_folder, "astroNN_0606_run001"))

        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)

        # assert temperature and gravity okay
        self.assertTrue(np.all(pred[0, 0:2] > [4700.0, 2.40]))
        self.assertTrue(np.all(pred[0, 0:2] < [4750.0, 2.47]))

    def test_arXiv_1902_08634(self):
        """
        astroNN spectrophotometric distance
        """
        from astroNN.apogee import visit_spectra, apogee_continuum
        from astroNN.gaia import extinction_correction, fakemag_to_pc
        from astropy.io import fits

        # first model
        models_url = [
            "https://github.com/henrysky/astroNN_gaia_dr2_paper/trunk/astroNN_no_offset_model",
            "https://github.com/henrysky/astroNN_gaia_dr2_paper/trunk/astroNN_constant_model",
            "https://github.com/henrysky/astroNN_gaia_dr2_paper/trunk/astroNN_multivariate_model",
        ]
        download_models(models_url)

        opened_fits = fits.open(
            visit_spectra(dr=14, location=4405, apogee="2M19060637+4717296")
        )
        spectrum = opened_fits[1].data
        spectrum_err = opened_fits[2].data
        spectrum_bitmask = opened_fits[3].data

        # using default continuum and bitmask values to continuum normalize
        norm_spec, norm_spec_err = apogee_continuum(
            spectrum, spectrum_err, bitmask=spectrum_bitmask, dr=14
        )
        # correct for extinction
        K = extinction_correction(
            opened_fits[0].header["K"], opened_fits[0].header["AKTARG"]
        )

        # ===========================================================================================#
        # load neural net
        neuralnet = load_folder(os.path.join(ci_data_folder, "astroNN_no_offset_model"))
        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)
        # convert prediction in fakemag to distance
        pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err["total"][:, 0])
        # assert distance is close enough
        # http://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
        # no offset correction so further away
        self.assertTrue(pc.value < 1250)
        self.assertTrue(pc.value > 1100)

        # ===========================================================================================#
        # load neural net
        neuralnet = load_folder(os.path.join(ci_data_folder, "astroNN_constant_model"))
        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(
            np.hstack([norm_spec, np.zeros((norm_spec.shape[0], 4))])
        )
        # convert prediction in fakemag to distance
        pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err["total"][:, 0])
        # assert distance is close enough
        # http://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
        self.assertTrue(pc.value < 1150)
        self.assertTrue(pc.value > 1000)

        # ===========================================================================================#
        # load neural net
        neuralnet = load_folder(os.path.join(ci_data_folder, "astroNN_multivariate_model"))
        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(
            np.hstack([norm_spec, np.zeros((norm_spec.shape[0], 4))])
        )
        # convert prediction in fakemag to distance
        pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err["total"][:, 0])
        # assert distance is close enough
        # http://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
        self.assertTrue(pc.value < 1150)
        self.assertTrue(pc.value > 1000)

    def test_arXiv_2302_05479(self):
        """
        astroNN paper models for spectroscopic age with encoder-decoder
        """
        from astroNN.apogee import visit_spectra, apogee_continuum
        from astropy.io import fits

        # first model
        models_url = [
            "https://github.com/henrysky/astroNN_ages/trunk/models/astroNN_VEncoderDecoder"
        ]
        download_models(models_url)
        
        # load the trained encoder-decoder model with astroNN
        neuralnet = load_folder(os.path.join(ci_data_folder, "astroNN_VEncoderDecoder"))

        # arbitrary spectrum
        opened_fits = fits.open(
            visit_spectra(
                dr=17,
                field="K06_078+16",
                telescope="apo25m",
                apogee="2M19060637+4717296",
            )
        )
        spectrum = opened_fits[1].data[0]
        spectrum_err = opened_fits[2].data[0]
        spectrum_bitmask = opened_fits[3].data[0]

        # using default continuum and bitmask values to continuum normalize
        norm_spec, norm_spec_err = apogee_continuum(
            spectrum, spectrum_err, bitmask=spectrum_bitmask, dr=17
        )
        
        # take care of extreme value
        norm_spec[norm_spec > 2.0] = 1.0

        # PSD reconstruction for the spectra
        psd_reconstruction = np.exp(neuralnet.predict(norm_spec)[0])

        # sampled latent space representation of the APOGEE spectrum
        z = neuralnet.predict_encoder(norm_spec)[0]

        # PSD prediction from latent space
        psd_from_z = np.exp(neuralnet.predict_decoder(z)[0])
        
        # known value of the latent space vector of this stars for THIS PARTICULAR MODEL
        self.assertTrue(np.all(z < [0.06, -0.73, -0.35, 0.06, 0.90]) & np.all(z > [0.00, -0.85, -0.45, -0.02, 0.75]))
        
        # make sure reconstruction from input directly and prediction from latent space vector are close enough
        self.assertTrue(np.max(psd_reconstruction - psd_from_z) < 0.8)
