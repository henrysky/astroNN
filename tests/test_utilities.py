import os

import astroNN
import numpy as np
import numpy.testing as npt
import pytest
from astroNN.config import MAGIC_NUMBER, config_path
from astroNN.datasets.h5 import h5name_check
from astroNN.nn.utilities.normalizer import Normalizer
from astroNN.shared import pylab_style
from astroNN.shared.downloader_tools import filehash
from astroNN.shared.nn_tools import cpu_fallback


def test_pltstyle():
    pylab_style(paper=False)
    pylab_style(paper=True)


def test_checksum():
    test_data_path = os.path.join(
        os.path.dirname(astroNN.__path__[0]), "astroNN", "data", "dr17_contmask.npy"
    )
    md5_pred = filehash(test_data_path, algorithm="md5")
    sha1_pred = filehash(test_data_path, algorithm="sha1")
    sha256_pred = filehash(test_data_path, algorithm="sha256")

    # read answer hashed by Windows Get-FileHash
    # filehash always return lower case so just in case
    assert md5_pred == "a646a9707e7aa2d943417c7e603e3731".lower()
    assert sha1_pred == "f701087e845b12b43f87c0d49fd15597bac9f171".lower()
    assert (
        sha256_pred
        == "a5705443e33698547ff6f7d6145ff8a4b8b3051a425aef468490a06e233dadb1".lower()
    )
    with pytest.raises(ValueError):
        filehash(test_data_path, algorithm="sha123")


def test_normalizer():
    data = np.random.normal(0, 1, (100, 10))
    magic_idx = (10, 5)
    data[magic_idx] = MAGIC_NUMBER

    # create a normalizer instance for mode 0
    normer = Normalizer(mode=0)
    norm_data = normer.normalize(data)
    # make sure normalizer preserve magic_number
    npt.assert_equal(norm_data[magic_idx], MAGIC_NUMBER)
    # test demoralize
    data_denorm = normer.denormalize(norm_data)
    # make sure demoralizer preserve magic_number
    npt.assert_equal(data_denorm[magic_idx], MAGIC_NUMBER)
    npt.assert_array_almost_equal(data_denorm, data)
    npt.assert_array_almost_equal(norm_data, data)

    # create a normalizer instance for mode 1
    normer = Normalizer(mode=1)
    norm_data = normer.normalize(data)
    # make sure normalizer preserve magic_number
    npt.assert_equal(norm_data[magic_idx], MAGIC_NUMBER)
    # test demoralize
    data_denorm = normer.denormalize(norm_data)
    # make sure demoralizer preserve magic_number
    npt.assert_equal(data_denorm[magic_idx], MAGIC_NUMBER)
    npt.assert_array_almost_equal(data_denorm, data)

    # test mode='3s' can do identity transformation
    s3_norm = Normalizer(mode="3s")
    data = np.random.normal(0, 1, (100, 10))
    npt.assert_array_almost_equal(
        s3_norm.denormalize(s3_norm.normalize(data)), data, decimal=5
    )

    data_8bit = np.random.randint(0, 256, (100, 50, 50))
    normer = Normalizer(mode=255)
    norm_data_8bit = normer.normalize(data_8bit)
    # make sure max of normalized image is 1.
    assert np.max(norm_data_8bit) == 1.0
    # make sure max of normalized image is 0.
    assert np.min(norm_data_8bit) == 0.0

    normer = Normalizer(mode={"input": 255, "aux": 0})
    norm_data_dict = normer.normalize({"input": data_8bit, "aux": data})
    # make sure max of normalized image is 1.
    assert np.max(norm_data_dict["input"]) == 1.0
    # make sure min of normalized image is 0.
    assert np.min(norm_data_dict["input"]) == 0.0
    npt.assert_array_almost_equal(
        norm_data_dict["aux"], data
    )  # make sure aux data is not normalized in this case

    errorous_norm = Normalizer(mode=-1234)
    with pytest.raises(ValueError):
        errorous_norm.normalize(data)


def test_cpu_gpu_management():
    cpu_fallback(flag=True)
    cpu_fallback(flag=False)

    # make sure flag=2 raise error
    with pytest.raises(ValueError):
        cpu_fallback(flag=2)


def test_h5name_check():
    # make sure h5name=None raise error
    with pytest.raises(ValueError):
        h5name_check(None)


def test_config():
    config_path(flag=0)
    config_path(flag=1)
    config_path(flag=2)
