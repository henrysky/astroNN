import os
import pytest
import requests
import numpy as np
from astroNN.data import datapath, data_description
from astroNN.datasets import xmatch
import numpy as np
import importlib
import numpy.testing as npt


def test_xmatch():
    # Some coordinates for cat1, J2000.
    cat1_ra = np.array([36.0, 68.0, 105.0, 23.0, 96.0, 96.0])
    cat1_dec = np.array([72.0, 56.0, 54.0, 55.0, 88.0, 88.0])

    # Some coordinates for cat2, J2000.
    cat2_ra = np.array([23.0, 56.0, 222.0, 96.0, 245.0, 68.0])
    cat2_dec = np.array([36.0, 68.0, 82.0, 88.0, 26.0, 56.0])

    # Using maxdist=2 arcsecond separation threshold, because its default, so not shown here
    # Using epoch1=2000. and epoch2=2000., because its default, so not shown here
    # because both datasets are J2000., so no need to provide pmra and pmdec which represent proper motion
    idx_1, idx_2, sep = xmatch(
        ra1=cat1_ra, dec1=cat1_dec, ra2=cat2_ra, dec2=cat2_dec
    )
    assert len(idx_1) == len(idx_2)
    npt.assert_equal(sep, np.zeros_like(sep))

@pytest.mark.parametrize("module_name", ["galaxy10", "galaxy10sdss"])
def test_galaxy10(module_name):
    g10_module = importlib.import_module(f"astroNN.datasets.{module_name}")
    galaxy10cls_lookup = getattr(g10_module, "galaxy10cls_lookup")
    galaxy10_confusion = getattr(g10_module, "galaxy10_confusion")
    _G10_ORIGIN = getattr(g10_module, "_G10_ORIGIN")
    _filename = getattr(g10_module, "_filename")

    # make sure galaxy10 exists on server
    r = requests.head(_G10_ORIGIN + _filename, allow_redirects=True, verify=False)
    assert r.status_code == 200
    r.close()

    galaxy10cls_lookup(0)
    with pytest.raises(ValueError):
        # only 10 classes
        galaxy10cls_lookup(11)
    galaxy10_confusion(np.ones((10, 10)))

def test_data():
    os.path.isdir(datapath())
    data_description()
