import unittest

import numpy as np
from astroNN.lamost import wavelength_solution, pseudo_continuum


class LamostToolsTestCase(unittest.TestCase):
    def test_wavelength_solution(self):
        wavelength_solution()
        wavelength_solution(dr=5)
        self.assertRaises(ValueError, wavelength_solution, dr=1)

    def test_norm(self):
        pseudo_continuum(np.ones(3909), np.ones(3909))


if __name__ == '__main__':
    unittest.main()
