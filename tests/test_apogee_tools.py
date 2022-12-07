import unittest

import numpy as np
import numpy.testing as npt
from astroNN.apogee import (
    gap_delete,
    apogee_default_dr,
    bitmask_decompositor,
    chips_split,
    bitmask_boolean,
    apogee_continuum,
    aspcap_mask,
    combined_spectra,
    visit_spectra,
)
from astroNN.apogee.apogee_shared import apogeeid_digit


class ApogeeToolsCase(unittest.TestCase):
    def test_apogee_tools(self):
        # Example data
        raw_spectra = np.ones((10, 8575))
        raw_spectrum = np.ones(8575)
        wrong_spectrum = np.ones(1024)

        gap_deleted = gap_delete(raw_spectra)
        self.assertEqual(gap_deleted.shape == (10, 7514), True)
        gap_deleted = gap_delete(raw_spectrum)
        self.assertEqual(gap_deleted.shape == (1, 7514), True)
        gap_deleted = gap_delete(raw_spectra, dr=12)
        self.assertEqual(gap_deleted.shape == (10, 7214), True)
        gap_deleted = gap_delete(raw_spectrum, dr=12)
        self.assertEqual(gap_deleted.shape == (1, 7214), True)
        self.assertRaises(EnvironmentError, gap_delete, wrong_spectrum)

        # check gaia default dr
        dr = apogee_default_dr()
        self.assertEqual(dr, 17)
        dr = apogee_default_dr(dr=3)
        self.assertEqual(dr, 3)

        # bitmask
        self.assertEqual(bitmask_decompositor(0), None)
        npt.assert_array_equal(bitmask_decompositor(1), [0])
        npt.assert_array_equal(bitmask_decompositor(3), [0, 1])
        npt.assert_array_equal(bitmask_boolean([0, 1, 2], [0]), [[False, True, False]])
        self.assertRaises(ValueError, bitmask_decompositor, -1)

        # chips_split
        blue, green, red = chips_split(raw_spectra)
        self.assertEqual(
            np.concatenate((blue, green, red), axis=1).shape == (10, 7514), True
        )
        blue, green, red = chips_split(raw_spectrum)
        self.assertEqual(
            np.concatenate((blue, green, red), axis=1).shape == (1, 7514), True
        )
        self.assertRaises(ValueError, chips_split, raw_spectra, dr=10)

    def test_apogee_continuum(self):
        raw_spectra = np.ones((10, 8575)) * 2
        raw_spectra_err = np.zeros((10, 8575))
        # continuum
        cont_spectra, cont_spectra_arr = apogee_continuum(raw_spectra, raw_spectra_err)
        self.assertAlmostEqual(float(np.mean(cont_spectra)), 1.0)

    def test_apogee_digit_extractor(self):
        # Test apogeeid digit extractor
        # just to make no error
        apogeeid_digit(["2M00380508+5608579", "2M00380508+5608579"])
        apogeeid_digit(np.array(["2M00380508+5608579", "2M00380508+5608579"]))

        # check accuracy
        self.assertEqual(apogeeid_digit("2M00380508+5608579"), "2003805085608579")
        npt.assert_array_equal(
            apogeeid_digit(np.array(["2M00380508+5608579", "2M00380508+5608579"])),
            ["2003805085608579", "2003805085608579"],
        )

    def test_aspcap_mask(self):
        self.assertEqual(np.all(aspcap_mask("C1") == aspcap_mask("ci")), True)
        self.assertEqual(np.all(aspcap_mask("TIII") == aspcap_mask("ti2")), True)
        # assert for example dr=1 is not supported
        self.assertRaises(ValueError, aspcap_mask, "al", 1)
        # Make sure if element not found, the case is nicely handled
        self.assertEqual(aspcap_mask("abc"), None)


class ApogeeDownloaderCase(unittest.TestCase):
    def test_apogee_combined_download(self):
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
        self.assertEqual(
            combined_spectra(dr=13, location=4406, apogee="2M19060637+4717296"), False
        )
        self.assertEqual(
            combined_spectra(dr=14, location=4406, apogee="2M19060637+4717296"), False
        )
        self.assertEqual(
            combined_spectra(
                dr=16,
                field="K06_078+17",
                telescope="apo25m",
                apogee="2M19060637+4717296",
            ),
            False,
        )
        self.assertEqual(
            combined_spectra(
                dr=17,
                field="K06_078+17",
                telescope="apo25m",
                apogee="2M19060637+4717296",
            ),
            False,
        )
        # assert error if DR not supported
        self.assertRaises(
            ValueError,
            combined_spectra,
            dr=1,
            location=4406,
            apogee="2M19060637+4717296",
        )

    def test_apogee_visit_download(self):
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
        self.assertEqual(
            visit_spectra(dr=13, location=4406, apogee="2M19060637+4717296"), False
        )
        self.assertEqual(
            visit_spectra(dr=14, location=4406, apogee="2M19060637+4717296"), False
        )
        self.assertEqual(
            visit_spectra(
                dr=16,
                field="K06_078+17",
                telescope="apo25m",
                apogee="2M19060637+4717296",
            ),
            False,
        )
        self.assertEqual(
            visit_spectra(
                dr=17,
                field="K06_078+17",
                telescope="apo25m",
                apogee="2M19060637+4717296",
            ),
            False,
        )
        # assert error if DR not supported
        self.assertRaises(
            ValueError, visit_spectra, dr=1, location=4406, apogee="2M19060637+4717296"
        )


if __name__ == "__main__":
    unittest.main()
