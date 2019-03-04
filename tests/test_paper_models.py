##################################################################
##
## To make sure publish models in papers actually still work fine
##
##################################################################

import unittest
import subprocess

import numpy as np
from astroNN.models import load_folder


class PapersModelsCase(unittest.TestCase):
    """
    Make sure all paper model work correctly
    """
    def test_arXiv_1808_04428(self):
        """
        original astroNN paper models
        """
        from astroNN.apogee import visit_spectra, apogee_continuum
        from astropy.io import fits

        # first model
        models_url = ["https://github.com/henrysky/astroNN_spectra_paper_figures/trunk/astroNN_0606_run001",
                     "https://github.com/henrysky/astroNN_spectra_paper_figures/trunk/astroNN_0617_run001"]

        for model_url in models_url:
            download_args = ["svn", "export", model_url]
            res = subprocess.Popen(download_args, stdout=subprocess.PIPE)
            output, _error = res.communicate()
            if not _error:
                pass
            else:
                raise ConnectionError(f"Error downloading the models {model_url}")

        opened_fits = fits.open(visit_spectra(dr=14, location=4405, apogee='2M19060637+4717296'))
        spectrum = opened_fits[1].data
        spectrum_err = opened_fits[2].data
        spectrum_bitmask = opened_fits[3].data

        # using default continuum and bitmask values to continuum normalize
        norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                    bitmask=spectrum_bitmask, dr=14)
        # load neural net
        neuralnet = load_folder('astroNN_0617_run001')

        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)

        # assert temperature and gravity okay
        self.assertEqual(np.all(pred[0, 0:2] > [4700., 2.40]), True)
        self.assertEqual(np.all(pred[0, 0:2] < [4750., 2.47]), True)

        # load neural net
        neuralnet = load_folder('astroNN_0606_run001')

        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)

        # assert temperature and gravity okay
        self.assertEqual(np.all(pred[0, 0:2] > [4700., 2.40]), True)
        self.assertEqual(np.all(pred[0, 0:2] < [4750., 2.47]), True)

    def test_arXiv_1902_08634 (self):
        """
        astroNN spectrophotometric distance
        """
        from astroNN.apogee import visit_spectra, apogee_continuum
        from astroNN.gaia import extinction_correction, fakemag_to_pc
        from astropy.io import fits

        # first model
        models_url = ["https://github.com/henrysky/astroNN_gaia_dr2_paper/trunk/astroNN_no_offset_model",
                      "https://github.com/henrysky/astroNN_gaia_dr2_paper/trunk/astroNN_constant_model",
                      "https://github.com/henrysky/astroNN_gaia_dr2_paper/trunk/astroNN_multivariate_model"]

        for model_url in models_url:
            download_args = ["svn", "export", model_url]
            res = subprocess.Popen(download_args, stdout=subprocess.PIPE)
            output, _error = res.communicate()
            if not _error:
                pass
            else:
                raise ConnectionError(f"Error downloading the models {model_url}")

        opened_fits = fits.open(visit_spectra(dr=14, location=4405, apogee='2M19060637+4717296'))
        spectrum = opened_fits[1].data
        spectrum_err = opened_fits[2].data
        spectrum_bitmask = opened_fits[3].data

        # using default continuum and bitmask values to continuum normalize
        norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                    bitmask=spectrum_bitmask, dr=14)
        # correct for extinction
        K = extinction_correction(opened_fits[0].header['K'], opened_fits[0].header['AKTARG'])

        # ===========================================================================================#
        # load neural net
        neuralnet = load_folder('astroNN_no_offset_model')
        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(norm_spec)
        # convert prediction in fakemag to distance
        pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err['total'][:, 0])
        # assert distance is close enough
        # http://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
        # no offset correction so further away
        self.assertEqual(pc.value < 1250, True)
        self.assertEqual(pc.value > 1100, True)

        # ===========================================================================================#
        # load neural net
        neuralnet = load_folder('astroNN_constant_model')
        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(np.hstack([norm_spec, np.zeros((norm_spec.shape[0], 4))]))
        # convert prediction in fakemag to distance
        pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err['total'][:, 0])
        # assert distance is close enough
        # http://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
        self.assertEqual(pc.value < 1150, True)
        self.assertEqual(pc.value > 1000, True)

        # ===========================================================================================#
        # load neural net
        neuralnet = load_folder('astroNN_multivariate_model')
        # inference, if there are multiple visits, then you should use the globally
        # weighted combined spectra (i.e. the second row)
        pred, pred_err = neuralnet.test(np.hstack([norm_spec, np.zeros((norm_spec.shape[0], 4))]))
        # convert prediction in fakemag to distance
        pc, pc_error = fakemag_to_pc(pred[:, 0], K, pred_err['total'][:, 0])
        # assert distance is close enough
        # http://simbad.u-strasbg.fr/simbad/sim-id?mescat.distance=on&Ident=%406876647&Name=KIC+10196240&submit=display+selected+measurements#lab_meas
        self.assertEqual(pc.value < 1150, True)
        self.assertEqual(pc.value > 1000, True)
