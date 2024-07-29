import numpy as np
import numpy.testing as npt
import pytest
from astroNN.apogee import (
    apogee_continuum,
    apogee_default_dr,
    aspcap_mask,
    bitmask_boolean,
    bitmask_decompositor,
    chips_split,
    combined_spectra,
    gap_delete,
    visit_spectra,
)
from astroNN.apogee.apogee_shared import apogeeid_digit


def test_apogee_tools():
    # Example data
    raw_spectra = np.ones((10, 8575))
    raw_spectrum = np.ones(8575)
    wrong_spectrum = np.ones(1024)

    gap_deleted = gap_delete(raw_spectra)
    assert gap_deleted.shape == (10, 7514)
    gap_deleted = gap_delete(raw_spectrum)
    assert gap_deleted.shape == (1, 7514)
    gap_deleted = gap_delete(raw_spectra, dr=12)
    assert gap_deleted.shape == (10, 7214)
    gap_deleted = gap_delete(raw_spectrum, dr=12)
    assert gap_deleted.shape == (1, 7214)
    with pytest.raises(EnvironmentError):
        gap_delete(wrong_spectrum)

    # check APOGEE default dr
    dr = apogee_default_dr()
    assert dr == 17
    dr = apogee_default_dr(dr=3)
    assert dr == 3

    # bitmask
    assert bitmask_decompositor(0) is None
    npt.assert_array_equal(bitmask_decompositor(1), [0])
    npt.assert_array_equal(bitmask_decompositor(3), [0, 1])
    npt.assert_array_equal(bitmask_boolean([0, 1, 2], [0]), [[False, True, False]])
    with pytest.raises(ValueError):
        bitmask_decompositor(-1)

    # chips_split
    blue, green, red = chips_split(raw_spectra)
    assert np.concatenate((blue, green, red), axis=1).shape == (10, 7514)
    blue, green, red = chips_split(raw_spectrum)
    assert np.concatenate((blue, green, red), axis=1).shape == (1, 7514)
    with pytest.raises(ValueError):
        chips_split(raw_spectra, dr=10)


def test_apogee_continuum():
    raw_spectra = np.ones((10, 8575)) * 2
    raw_spectra_err = np.zeros((10, 8575))
    # continuum
    cont_spectra, cont_spectra_arr = apogee_continuum(raw_spectra, raw_spectra_err)
    npt.assert_almost_equal(float(np.mean(cont_spectra)), 1.0)


def test_apogee_digit_extractor():
    # Test apogeeid digit extractor
    # just to make no error
    apogeeid_digit(["2M00380508+5608579", "2M00380508+5608579"])
    apogeeid_digit(np.array(["2M00380508+5608579", "2M00380508+5608579"]))

    # check accuracy
    assert apogeeid_digit("2M00380508+5608579") == "2003805085608579"
    npt.assert_array_equal(
        apogeeid_digit(np.array(["2M00380508+5608579", "2M00380508+5608579"])),
        ["2003805085608579", "2003805085608579"],
    )


def test_aspcap_mask():
    assert np.all(aspcap_mask("C1") == aspcap_mask("ci"))
    assert np.all(aspcap_mask("TIII") == aspcap_mask("ti2"))
    # assert for example dr=1 is not supported
    with pytest.raises(ValueError):
        aspcap_mask("al", 1)
    # Make sure if element not found, the case is nicely handled
    with pytest.raises(ValueError):
        aspcap_mask("abc", 1)


def test_apogee_combined_download():
    """
    Test APOGEE combined spectra downloading function, assert functions can deal with missing files
    """
    # make sure the download works correctly
    combined_spectra(dr=13, location=4405, apogee="2M19060637+4717296")
    combined_spectra(dr=14, location=4405, apogee="2M19060637+4717296")
    combined_spectra(
        dr=16, field="K06_078+16", telescope="apo25m", apogee="2M19060637+4717296"
    )
    combined_spectra(
        dr=17, field="K06_078+16", telescope="apo25m", apogee="2M19060637+4717296"
    )
    # assert False is returning if file not found
    assert combined_spectra(dr=13, location=4406, apogee="2M19060637+4717296") is False
    assert combined_spectra(dr=14, location=4406, apogee="2M19060637+4717296") is False

    assert (
        combined_spectra(
            dr=16, field="K06_078+17", telescope="apo25m", apogee="2M19060637+4717296"
        )
        is False
    )
    assert (
        combined_spectra(
            dr=17, field="K06_078+17", telescope="apo25m", apogee="2M19060637+4717296"
        )
        is False
    )
    # assert error if DR not supported
    with pytest.raises(ValueError):
        combined_spectra(dr=1, location=4406, apogee="2M19060637+4717296")


def test_apogee_visit_download():
    """
    Test APOGEE visits spectra downloading function, assert functions can deal with missing files
    """
    # make sure the download works correctly
    visit_spectra(dr=13, location=4405, apogee="2M19060637+4717296")
    visit_spectra(dr=14, location=4405, apogee="2M19060637+4717296")
    visit_spectra(
        dr=16, field="K06_078+16", telescope="apo25m", apogee="2M19060637+4717296"
    )
    visit_spectra(
        dr=17, field="K06_078+16", telescope="apo25m", apogee="2M19060637+4717296"
    )
    # assert False is returning if file not found
    assert visit_spectra(dr=13, location=4406, apogee="2M19060637+4717296") is False
    assert (
        visit_spectra(
            dr=16, field="K06_078+17", telescope="apo25m", apogee="2M19060637+4717296"
        )
        is False
    )
    assert (
        visit_spectra(
            dr=17, field="K06_078+17", telescope="apo25m", apogee="2M19060637+4717296"
        )
        is False
    )
    # assert error if DR not supported
    with pytest.raises(ValueError):
        visit_spectra(dr=1, location=4406, apogee="2M19060637+4717296")
