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
    original astroNN paper models
    """
    def test_arXiv_1808_04428(self):
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
                raise ConnectionError("Error downloading the model")

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