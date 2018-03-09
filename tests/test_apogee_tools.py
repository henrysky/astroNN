import unittest
import numpy as np
import numpy.testing as npt
import astropy.units as u
from astroNN.apogee import gap_delete


class GaiaToolsCase(unittest.TestCase):
    def test_astrometry_conversion(self):
        # Example data
        raw_spectra = np.ones((10, 8575))
        raw_spectrum = np.ones((8575))

        gap_deleted = gap_delete(raw_spectra)
        self.assertEquals(gap_deleted.shape == (10, 7514), True)

        gap_deleted = gap_delete(raw_spectrum)
        self.assertEquals(gap_deleted.shape == (1, 7514), True)


if __name__ == '__main__':
    unittest.main()
