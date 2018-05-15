import unittest
from astroNN.lamost import wavelength_solution

class LamostToolsTestCase(unittest.TestCase):
    def test_wavelength_solution(self):
        wavelength_solution()
        wavelength_solution(dr=5)
        wavelength_solution(dr=1)


if __name__ == '__main__':
    unittest.main()
