import os
import urllib.request

import h5py
import keras
import numpy as np
import pytest
import requests

from astroNN.shared.downloader_tools import TqdmUpTo


@pytest.fixture(scope="session")
def spectra_ci_data():
    _URL_ORIGIN = "https://www.astro.utoronto.ca/~hleung/shared/ci_data/"
    filename = "apogee_dr14_green_nan.h5"
    complete_url = _URL_ORIGIN + filename
    if not os.path.exists("ci_data"):
        os.mkdir("ci_data")
    local_file_path = os.path.join("ci_data", filename)

    # Check if files exists
    if not os.path.isfile(local_file_path):
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=complete_url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                complete_url, local_file_path, reporthook=t.update_to
            )
    else:
        r = requests.head(complete_url, allow_redirects=True, verify=True)
        assert r.status_code == 200, f"CI data file does not exist on {complete_url}"

    # Data preparation
    f = h5py.File(local_file_path, "r")
    xdata = np.asarray(f["spectra"])
    ydata = np.stack([f["logg"], f["feh"]]).T
    ydata_err = np.stack([f["logg_err"], f["feh_err"]]).T
    return xdata, ydata, ydata_err


@pytest.fixture(scope="session")
def mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    # To convert to desirable type
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    return x_train, y_train, x_test, y_test
