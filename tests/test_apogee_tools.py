import unittest
import numpy as np
import numpy.testing as npt
from astroNN.apogee import gap_delete, apogee_default_dr, bitmask_decompositor, chips_split, bitmask_boolean


class GaiaToolsCase(unittest.TestCase):
    def test_astrometry_conversion(self):
        # Example data
        raw_spectra = np.ones((10, 8575))
        raw_spectrum = np.ones((8575))

        gap_deleted = gap_delete(raw_spectra)
        self.assertEquals(gap_deleted.shape == (10, 7514), True)
        gap_deleted = gap_delete(raw_spectrum)
        self.assertEquals(gap_deleted.shape == (1, 7514), True)
        gap_deleted = gap_delete(raw_spectra, dr=12)
        self.assertEquals(gap_deleted.shape == (10, 7214), True)
        gap_deleted = gap_delete(raw_spectrum, dr=12)
        self.assertEquals(gap_deleted.shape == (1, 7214), True)

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

        # chips_split
        blue, green, red = chips_split(raw_spectra)
        self.assertEquals(np.concatenate((blue, green, red), axis=1).shape == (10, 7514), True)
        blue, green, red = chips_split(raw_spectrum)
        self.assertEquals(np.concatenate((blue, green, red), axis=1).shape == (1, 7514), True)


if __name__ == '__main__':
    unittest.main()
