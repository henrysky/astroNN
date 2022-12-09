##################################################################
##
## To make sure publish models in papers actually still work fine
##
##################################################################

import unittest
import subprocess

import numpy as np
from astroNN.models import load_folder


def download_models(models_url):
    """
    function to download model directly from github url
    """
    for model_url in models_url:
        download_args = ["svn", "export", model_url]
        res = subprocess.Popen(download_args, stdout=subprocess.PIPE)
        output, _error = res.communicate()
        if not _error:
            pass
        else:
            raise ConnectionError(f"Error downloading the models {model_url}")


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
        neuralnet = load_folder("astroNN_0617_run001")

        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)

        # assert temperature and gravity okay
        self.assertTrue(np.all(pred[0, 0:2] > [4700.0, 2.40]))
        self.assertTrue(np.all(pred[0, 0:2] < [4750.0, 2.47]))

        # load neural net
        neuralnet = load_folder("astroNN_0606_run001")

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
        neuralnet = load_folder("astroNN_no_offset_model")
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
        neuralnet = load_folder("astroNN_constant_model")
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
        neuralnet = load_folder("astroNN_multivariate_model")
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

    def test_arXiv_pending(self):
        """
        astroNN paper models for spectroscopic age with encoder-decoder
        """
        from astroNN.apogee import visit_spectra, apogee_continuum
        from astropy.io import fits

        # first model
        models_url = [
            "https://github.com/henrysky/astroNN_ages/trunk/astroNN_VEncoderDecoder"
        ]
        download_models(models_url)
        
        # TODO: temporary until we have submitted the paper
        download_args = ["wget", "-r", "-nH", "--cut-dirs=2", "--no-parent", "https://www.astro.utoronto.ca/~hleung/shared/astroNN_VEncoderDecoder/"]
        res = subprocess.Popen(download_args, stdout=subprocess.PIPE)
        output, _error = res.communicate()
        if not _error:
            pass
        else:
            raise ConnectionError("Error downloading the models")
        
        # load the trained encoder-decoder model with astroNN
        neuralnet = load_folder("astroNN_VEncoderDecoder")

        # arbitrary spectrum
        opened_fits = fits.open(
            visit_spectra(
                dr=17,
                field="K06_078+16",
                telescope="apo25m",
                apogee="2M19060637+4717296",
            )
        )
        spectrum = opened_fits[1].data
        spectrum_err = opened_fits[2].data
        spectrum_bitmask = opened_fits[3].data

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
        print(z)
        self.assertTrue(np.all(z < [0.31, -0.59, 1.05, -0.35, -1.05]) & np.all(z > [0.29, -0.61, 1.03, -0.38, -1.08]))
        
        # make sure reconstruction from input directly and prediction from latent space vector are close enough
        self.assertTrue(np.max(psd_reconstruction - psd_from_z) < 0.5)
