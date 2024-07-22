##################################################################
##
## To make sure publish models in papers actually still work fine
##
##################################################################

import os
import warnings
import unittest
import shutil
import subprocess

import numpy as np
from astroNN.models import load_folder

ci_data_folder = "ci_data"
if not os.path.exists(ci_data_folder):
    os.mkdir(ci_data_folder)


def download_models(repository_urls, folder_name):
    """
    function to download model directly from github url
    """
    repo_folder = repository_urls.split("/")[-1]
    if not os.path.exists(os.path.join(ci_data_folder, folder_name)):
        download_args = ["git", "clone", "-n", "--depth=1", "--filter=tree:0", repository_urls]
        res = subprocess.Popen(download_args, stdout=subprocess.PIPE)
        output, _error = res.communicate()

        checkout_args = ["git", "sparse-checkout", "set", "--no-cone", folder_name]
        res = subprocess.Popen(checkout_args, stdout=subprocess.PIPE, cwd=repo_folder)
        output, _error = res.communicate()

        checkout_args = ["git", "checkout"]
        res = subprocess.Popen(checkout_args, stdout=subprocess.PIPE, cwd=repo_folder)
        output, _error = res.communicate()

        if not _error:
            pass
        else:
            raise ConnectionError(f"Error downloading the models {folder_name} from {repository_urls}")
        
        shutil.move(os.path.join(repo_folder, folder_name), os.path.join(ci_data_folder, folder_name))
        shutil.rmtree(repo_folder)
    else:  # if the model is cached on Github Action, do a sanity check on remote folder without downloading it
        warnings.warn(f"Folder {folder_name} already exists, skipping download")


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
            {"repository_urls": "https://github.com/henrysky/astroNN_spectra_paper_figures", "folder_name": "astroNN_0606_run001"},
            {"repository_urls": "https://github.com/henrysky/astroNN_spectra_paper_figures", "folder_name": "astroNN_0617_run001"},
        ]
        for url in models_url:
            download_models(**url)

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
            {"repository_urls": "https://github.com/henrysky/astroNN_gaia_dr2_paper", "folder_name": "astroNN_no_offset_model"},
            {"repository_urls": "https://github.com/henrysky/astroNN_gaia_dr2_paper", "folder_name": "astroNN_constant_model"},
            {"repository_urls": "https://github.com/henrysky/astroNN_gaia_dr2_paper", "folder_name": "astroNN_multivariate_model"},
        ]
        for url in models_url:
            download_models(**url)

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
        # https://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
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
        # https://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
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
        # https://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
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
            {"repository_urls": "https://github.com/henrysky/astroNN_ages", "folder_name": "models"},
        ]
        for url in models_url:
            download_models(**url)
                
        # load the trained encoder-decoder model with astroNN
        neuralnet = load_folder(os.path.join(ci_data_folder, "models/astroNN_VEncoderDecoder"))

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
