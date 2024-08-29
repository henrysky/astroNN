import pathlib

import h5py
import keras
import numpy as np
import pytest
import requests


@pytest.fixture(scope="module")
def spectra_ci_data():
    _URL_ORIGIN = "https://www.astro.utoronto.ca/~hleung/shared/ci_data/"
    filename = "apogee_dr14_green_nan.h5"
    complete_url = _URL_ORIGIN + filename

    local_file_path = pathlib.Path("ci_data").absolute().joinpath(filename)
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if files exists
    if local_file_path.exists():
        # if file exists, check if the hosting server still has the file
        r = requests.head(complete_url, allow_redirects=True, verify=True)
        assert r.status_code == 200, f"CI data file does not exist on {complete_url}"
    else:
        # if file does not exist, download the file using requests
        r = requests.get(complete_url, allow_redirects=True, verify=True)

    # Data preparation
    with h5py.File(local_file_path, "r") as f:
        xdata = np.asarray(f["spectra"])
        ydata = np.stack([f["logg"], f["feh"]]).T
        ydata_err = np.stack([f["logg_err"], f["feh_err"]]).T
        return xdata, ydata, ydata_err


@pytest.fixture(scope="module")
def mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    return x_train, y_train, x_test, y_test
