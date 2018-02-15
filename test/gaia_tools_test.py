import unittest
import numpy as np
import numpy.testing as npt
from astroNN.gaia import absmag_to_pc


class GaiaToolsCase(unittest.TestCase):
    def test_astrometry_conversion(self):
        # Example data of [Vega, Sirius, Betelgeuse]
        absmag = np.array([0.582, 1.42, -5.85])
        mag = np.array([0.03, -1.46, 0.5])
        pc = absmag_to_pc(absmag, mag)
        npt.assert_almost_equal(pc.value, [7.8, 2.7, 186.2], decimal=1)


if __name__ == '__main__':
    unittest.main()
