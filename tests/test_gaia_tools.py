import unittest

import astropy.units as u
import numpy as np
import numpy.testing as npt

from astroNN.gaia import absmag_to_pc, mag_to_absmag, fakemag_to_absmag, absmag_to_fakemag, fakemag_to_pc, \
    mag_to_fakemag, gaia_default_dr


class GaiaToolsCase(unittest.TestCase):
    def test_astrometry_conversion(self):
        # Example data of [Vega, Sirius]
        absmag = np.array([0.582, 1.42])
        mag = np.array([0.026, -1.46])
        parallax = np.array([130.23, 379.21])
        parallax_err = np.array([0.36, 1.58])

        # Absmag related test
        pc = absmag_to_pc(absmag, mag)
        npt.assert_almost_equal(pc.value, 1000 / parallax, decimal=1)
        npt.assert_almost_equal(mag_to_absmag(mag, parallax * u.mas), absmag, decimal=1)
        absmag_test, absmag_err_test = mag_to_absmag(mag, parallax * u.mas, parallax_err)
        absmag_test_arc, absmag_err_test_arc = mag_to_absmag(mag, parallax / 1000 * u.arcsec, parallax_err / 1000)
        absmag_test_unitless, absmag_err_test_unitless = mag_to_absmag(mag, parallax, parallax_err)

        # make sure unitless same as using astropy unit
        npt.assert_almost_equal(absmag_test, absmag_test_unitless)
        npt.assert_almost_equal(absmag_err_test, absmag_err_test_unitless)

        # make sure astropy unit conversion works fine
        npt.assert_almost_equal(absmag_test, absmag_test_arc)
        npt.assert_almost_equal(absmag_err_test, absmag_err_test_arc)

        # =================== Fakemag related test ===================#
        # make sure these function did identity transform
        npt.assert_almost_equal(fakemag_to_absmag(absmag_to_fakemag(absmag)), absmag, decimal=1)

        # we can tests this after identity transformation confirmed
        npt.assert_almost_equal(fakemag_to_pc(absmag_to_fakemag(absmag), mag).value, 1000 / parallax, decimal=1)
        npt.assert_almost_equal(fakemag_to_pc(mag_to_fakemag(mag, parallax * u.mas), mag).value, 1000 / parallax,
                                decimal=1)

        fakemag_test, fakemag_err_test = mag_to_fakemag(mag, parallax * u.mas, parallax_err)
        fakemag_test_unitless, fakemag_err_test_unitless = mag_to_fakemag(mag, parallax, parallax_err)
        fakemag_test_arc, fakemag_err_test_arc = mag_to_fakemag(mag, parallax / 1000 * u.arcsec, parallax_err / 1000)

        pc_result, pc_result_err = fakemag_to_pc(fakemag_test, mag, fakemag_err_test)
        pc_result_arc, pc_result_err_arc = fakemag_to_pc(fakemag_test_arc, mag, fakemag_err_test_arc)

        # Analytically solution checkung
        npt.assert_almost_equal(pc_result_err.value, (parallax_err / parallax) * pc.value, decimal=1)

        # make sure unitless same as using astropy unit
        npt.assert_almost_equal(fakemag_test, fakemag_test_unitless)
        npt.assert_almost_equal(fakemag_err_test, fakemag_err_test_unitless)

        # make sure astropy unit conversion works fine
        npt.assert_almost_equal(fakemag_test, fakemag_test_arc)
        npt.assert_almost_equal(fakemag_err_test, fakemag_err_test_arc)
        npt.assert_almost_equal(pc_result.value, pc_result_arc.value)
        npt.assert_almost_equal(pc_result_err.value, pc_result_err_arc.value)

        # check gaia default dr
        dr = gaia_default_dr()
        self.assertEqual(dr, 1)
        dr = gaia_default_dr(dr=3)
        self.assertEqual(dr, 3)

    def test_anderson(self):
        from astroNN.gaia import anderson_2017_parallax
        # To load the improved parallax
        # Both parallax and para_var is in mas
        # cuts=True to cut bad data (negative parallax and percentage error more than 20%)
        ra, dec, parallax, para_err = anderson_2017_parallax(cuts=True)


if __name__ == '__main__':
    unittest.main()
