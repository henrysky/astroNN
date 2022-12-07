import unittest

import requests
import numpy as np
from astroNN.data import datapath, data_description
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
        idx_1, idx_2, sep = xmatch(ra1=cat1_ra, dec1=cat1_dec, ra2=cat2_ra, dec2=cat2_dec)
        self.assertEqual(len(idx_1), len(idx_2))
        self.assertEqual(np.all(sep==0.), True)
        
    def test_apokasc(self):
        from astroNN.datasets.apogee import load_apokasc

        ra, dec, logg = load_apokasc()
        gold_ra, gold_dec, gold_logg, basic_ra, basic_dec, basic_logg = load_apokasc(combine=False)

    def test_galaxy10(self):
        from astroNN.datasets.galaxy10 import galaxy10cls_lookup, galaxy10_confusion
        # make sure galaxy10 exists on astro's server

        r = requests.head(_G10_ORIGIN, allow_redirects=True, verify=False)
        self.assertEqual(r.status_code, 200)
        r.close()

        galaxy10cls_lookup(0)
        self.assertRaises(ValueError, galaxy10cls_lookup, 11)
        galaxy10_confusion(np.ones((10,10)))
        
    def test_galaxy10sdss(self):
        from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup, galaxy10_confusion
        # make sure galaxy10 exists on astro's server

        r = requests.head(_G10_ORIGIN, allow_redirects=True, verify=False)
        self.assertEqual(r.status_code, 200)
        r.close()

        galaxy10cls_lookup(0)
        self.assertRaises(ValueError, galaxy10cls_lookup, 11)
        galaxy10_confusion(np.ones((10,10)))

    def test_data(self):
        import os
        os.path.isdir(datapath())
        data_description()


if __name__ == '__main__':
    unittest.main()
