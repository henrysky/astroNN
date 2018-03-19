import unittest

import numpy as np
import numpy.testing as npt

from astroNN.apogee import gap_delete, apogee_default_dr, bitmask_decompositor, chips_split, bitmask_boolean, \
    apogee_continuum
from astroNN.apogee.apogee_shared import apogeeid_digit


class ApogeeToolsCase(unittest.TestCase):
    def test_apogee_tools(self):
        # Example data
        raw_spectra = np.ones((10, 8575))
        raw_spectrum = np.ones(8575)
        wrong_spectrum = np.ones(1024)

        gap_deleted = gap_delete(raw_spectra)
        self.assertEqual(gap_deleted.shape == (10, 7514), True)
        gap_deleted = gap_delete(raw_spectrum)
        self.assertEqual(gap_deleted.shape == (1, 7514), True)
        gap_deleted = gap_delete(raw_spectra, dr=12)
        self.assertEqual(gap_deleted.shape == (10, 7214), True)
        gap_deleted = gap_delete(raw_spectrum, dr=12)
        self.assertEqual(gap_deleted.shape == (1, 7214), True)
        self.assertRaises(EnvironmentError, gap_delete, wrong_spectrum)

        # check gaia default dr
        dr = apogee_default_dr()
        self.assertEqual(dr, 14)
        dr = apogee_default_dr(dr=3)
        self.assertEqual(dr, 3)

        # bitmask
        self.assertEqual(bitmask_decompositor(0), None)
        npt.assert_array_equal(bitmask_decompositor(1), [0])
        npt.assert_array_equal(bitmask_decompositor(3), [0, 1])
        npt.assert_array_equal(bitmask_boolean([0, 1, 2], [1]), [[True, True, False]])
        self.assertRaises(ValueError, bitmask_decompositor, -1)

        # chips_split
        blue, green, red = chips_split(raw_spectra)
        self.assertEqual(np.concatenate((blue, green, red), axis=1).shape == (10, 7514), True)
        blue, green, red = chips_split(raw_spectrum)
        self.assertEqual(np.concatenate((blue, green, red), axis=1).shape == (1, 7514), True)
        self.assertRaises(ValueError, lambda: chips_split(raw_spectra, dr=10))

        # Test apogeeid digit extractor
        self.assertEqual(apogeeid_digit("2M00380508+5608579"), '2003805085608579')

    def test_apogee_continuum(self):
        raw_spectra = np.ones((10, 8575))
        raw_spectra_err = np.zeros((10, 8575))
        # continuum
        cont_spectra, cont_spectra_arr = apogee_continuum(raw_spectra, raw_spectra_err)
        self.assertAlmostEqual(np.mean(cont_spectra), 1.)


if __name__ == '__main__':
    unittest.main()
