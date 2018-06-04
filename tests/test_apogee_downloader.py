import unittest

import numpy as np
import numpy.testing as npt
from astroNN.apogee import combined_spectra, visit_spectra


class ApogeeDownloaderCase(unittest.TestCase):
    def test_apogee_combined_download(self):
        combined_spectra(dr=13, location=4405, apogee='2M19060637+4717296')
        combined_spectra(dr=14, location=4405, apogee='2M19060637+4717296')
        self.assertEqual(combined_spectra(dr=13, location=4406, apogee='2M19060637+4717296'), False)
        self.assertEqual(combined_spectra(dr=14, location=4406, apogee='2M19060637+4717296'), False)
        self.assertRaises(ValueError, combined_spectra, dr=1, location=4406, apogee='2M19060637+4717296')

    def test_apogee_visit_download(self):
        visit_spectra(dr=13, location=4405, apogee='2M19060637+4717296')
        visit_spectra(dr=14, location=4405, apogee='2M19060637+4717296')
        self.assertEqual(visit_spectra(dr=13, location=4406, apogee='2M19060637+4717296'), False)
        self.assertEqual(visit_spectra(dr=14, location=4406, apogee='2M19060637+4717296'), False)
        self.assertRaises(ValueError, visit_spectra, dr=1, location=4406, apogee='2M19060637+4717296')


if __name__ == '__main__':
    unittest.main()
