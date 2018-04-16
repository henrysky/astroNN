import unittest
import requests
from astroNN.datasets.galaxy10 import _G10_ORIGIN


class DatasetTestCase(unittest.TestCase):
    def test_xmatch(self):
        from astroNN.datasets import xmatch
        import numpy as np

        # Some coordinates for cat1, J2000.
        cat1_ra = np.array([36., 68., 105., 23., 96., 96.])
        cat1_dec = np.array([72., 56., 54., 55., 88., 88.])

        # Some coordinates for cat2, J2000.
        cat2_ra = np.array([23., 56., 222., 96., 245., 68.])
        cat2_dec = np.array([36., 68., 82., 88., 26., 56.])

        # Using maxdist=2 arcsecond separation threshold, because its default, so not shown here
        # Using epoch1=2000. and epoch2=2000., because its default, so not shown here
        # because both datasets are J2000., so no need to provide pmra and pmdec which represent proper motion
        idx_1, idx_2, sep = xmatch(cat1_ra, cat2_ra, colRA1=cat1_ra, colDec1=cat1_dec, colRA2=cat2_ra, colDec2=cat2_dec,
                                   swap=False)
        self.assertEqual(len(idx_1), len(idx_2))

    def test_apokasc(self):
        from astroNN.datasets.apokasc import apokasc_load

        ra, dec, logg = apokasc_load()
        gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg = apokasc_load(combine=False)

    def test_galaxy10(self):
        # make sure galaxy10 exists on Bovy's server

        r = requests.head(_G10_ORIGIN, allow_redirects=True)
        self.assertEqual(r.status_code, 200)
        r.close()


if __name__ == '__main__':
    unittest.main()
