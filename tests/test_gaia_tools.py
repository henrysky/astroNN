import unittest

import astropy.units as u
import numpy as np
import numpy.testing as npt
from astroNN.config import MAGIC_NUMBER
from astroNN.gaia import absmag_to_pc, mag_to_absmag, fakemag_to_absmag, absmag_to_fakemag, fakemag_to_pc, \
    mag_to_fakemag, gaia_default_dr, fakemag_to_parallax, fakemag_to_mag


# noinspection PyUnresolvedReferences
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
        self.assertEqual(np.any(absmag_to_fakemag(MAGIC_NUMBER) == MAGIC_NUMBER), True)
        absmag_test, absmag_err_test = mag_to_absmag(mag, parallax * u.mas, parallax_err)
        absmag_test_uniterr, absmag_err_test_uniterr = mag_to_absmag(mag, parallax * u.mas,
                                                                     parallax_err / 1000 * u.arcsec)
        absmag_test_arc, absmag_err_test_arc = mag_to_absmag(mag, parallax / 1000 * u.arcsec, parallax_err / 1000)
        absmag_test_unitless, absmag_err_test_unitless = mag_to_absmag(mag, parallax, parallax_err)

        # make sure unitless same as using astropy unit
        npt.assert_almost_equal(absmag_test, absmag_test_unitless)
        npt.assert_almost_equal(absmag_test, absmag_test_unitless)
        npt.assert_almost_equal(absmag_err_test, absmag_err_test_uniterr)

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
        npt.assert_almost_equal(fakemag_to_mag(mag_to_fakemag(10, 1), 1000), 10.)

        fakemag_test, fakemag_err_test = mag_to_fakemag(mag, parallax * u.mas, parallax_err)
        fakemag_test_uniterr, fakemag_err_test_uniterr = mag_to_fakemag(mag, parallax * u.mas,
                                                                        parallax_err / 1000 * u.arcsec)
        fakemag_test_unitless, fakemag_err_test_unitless = mag_to_fakemag(mag, parallax, parallax_err)
        fakemag_test_arc, fakemag_err_test_arc = mag_to_fakemag(mag, parallax / 1000 * u.arcsec, parallax_err / 1000)

        pc_result, pc_result_err = fakemag_to_pc(fakemag_test, mag, fakemag_err_test)
        parallax_result, parallax_result_err = fakemag_to_parallax(fakemag_test, mag, fakemag_err_test)
        pc_result_arc, pc_result_err_arc = fakemag_to_pc(fakemag_test_arc, mag, fakemag_err_test_arc)

        # Analytically solution checkung
        npt.assert_almost_equal(pc_result_err.value, (parallax_err / parallax) * pc.value, decimal=1)

        # make sure unitless same as using astropy unit
        npt.assert_almost_equal(fakemag_test, fakemag_test_unitless)
        npt.assert_almost_equal(fakemag_err_test, fakemag_err_test_uniterr)
        npt.assert_almost_equal(fakemag_err_test, fakemag_err_test_unitless)

        # make sure astropy unit conversion works fine
        npt.assert_almost_equal(fakemag_test, fakemag_test_arc)
        npt.assert_almost_equal(fakemag_err_test, fakemag_err_test_arc)
        npt.assert_almost_equal(pc_result.value, pc_result_arc.value)
        npt.assert_almost_equal(pc_result.value, 1000 / parallax_result.value)
        npt.assert_almost_equal(pc_result_err.value, pc_result_err_arc.value)
        npt.assert_almost_equal(fakemag_to_mag(mag_to_fakemag(10, 1*u.mas), 1000*u.parsec), 10.)

        # check gaia default dr
        dr = gaia_default_dr()
        self.assertEqual(dr, 2)
        dr = gaia_default_dr(dr=3)
        self.assertEqual(dr, 3)

    def test_logsol(self):
        # Test conversion tools related to log solar luminosity
        from astroNN.gaia import fakemag_to_logsol, absmag_to_logsol, logsol_to_absmag, logsol_to_fakemag
        self.assertEqual(logsol_to_fakemag(fakemag_to_logsol(100.)), 100.)
        npt.assert_array_equal(logsol_to_fakemag(fakemag_to_logsol([100., 100.])), [100., 100.])
        npt.assert_array_equal(logsol_to_fakemag(fakemag_to_logsol(np.array([100, 100, 100]))), [100., 100., 100.])
        self.assertEqual(fakemag_to_logsol(MAGIC_NUMBER), MAGIC_NUMBER)
        self.assertEqual(logsol_to_fakemag(fakemag_to_logsol(MAGIC_NUMBER)), MAGIC_NUMBER)
        self.assertEqual(np.any(fakemag_to_logsol([MAGIC_NUMBER, 1000]) == MAGIC_NUMBER), True)

        self.assertEqual(logsol_to_absmag(absmag_to_logsol(99.)), 99.)
        self.assertAlmostEqual(logsol_to_absmag(absmag_to_logsol(-99.)), -99.)
        npt.assert_array_equal(logsol_to_absmag(absmag_to_logsol([99., 99.])), [99., 99.])
        npt.assert_array_almost_equal(logsol_to_absmag(absmag_to_logsol([-99., -99.])), [-99., -99.])
        npt.assert_array_almost_equal(logsol_to_absmag(absmag_to_logsol(np.array([99., 99., 99.]))), [99., 99., 99.])
        self.assertEqual(absmag_to_logsol(MAGIC_NUMBER), MAGIC_NUMBER)
        self.assertEqual(logsol_to_absmag(absmag_to_logsol(MAGIC_NUMBER)), MAGIC_NUMBER)
        self.assertEqual(np.any(absmag_to_logsol([MAGIC_NUMBER, 1000]) == MAGIC_NUMBER), True)

    def test_extinction(self):
        from astroNN.gaia import extinction_correction

        self.assertEqual(np.any([extinction_correction(10., -90.) == -9999.]), False)
        self.assertEqual(extinction_correction(10., -90.), 10.)
        self.assertEqual(np.any([extinction_correction(-99.99, -90.) == -9999.]), True)

    def test_known_regression(self):
        # prevent regression of known bug
        self.assertEqual(mag_to_absmag(1., MAGIC_NUMBER), MAGIC_NUMBER)
        self.assertEqual(mag_to_absmag(MAGIC_NUMBER, MAGIC_NUMBER), MAGIC_NUMBER)
        self.assertEqual(
            np.all(mag_to_absmag(MAGIC_NUMBER, MAGIC_NUMBER, 1.) == (MAGIC_NUMBER, MAGIC_NUMBER)), True)
        self.assertEqual(
            np.all(mag_to_fakemag(MAGIC_NUMBER, MAGIC_NUMBER, 1.) == (MAGIC_NUMBER, MAGIC_NUMBER)), True)

        self.assertEqual(mag_to_fakemag(1., MAGIC_NUMBER), MAGIC_NUMBER)
        self.assertEqual(mag_to_fakemag(MAGIC_NUMBER, MAGIC_NUMBER), MAGIC_NUMBER)

        self.assertEqual(fakemag_to_pc(1., MAGIC_NUMBER).value, MAGIC_NUMBER)
        self.assertEqual(fakemag_to_pc(-1., 2.).value, MAGIC_NUMBER)
        self.assertEqual(absmag_to_pc(1., MAGIC_NUMBER).value, MAGIC_NUMBER)

    def test_anderson(self):
        from astroNN.gaia import anderson_2017_parallax
        # To load the improved parallax
        # Both parallax and para_var is in mas
        # cuts=True to cut bad data (negative parallax and percentage error more than 20%)
        ra, dec, parallax, para_err = anderson_2017_parallax(cuts=True)
        self.assertEqual(np.any([parallax == -9999.]), False)  # assert no -9999

    def test_dr2_parallax(self):
        from astroNN.gaia import gaiadr2_parallax
        # To load the improved parallax
        # Both parallax and para_var is in mas
        # cuts=True to cut bad data (negative parallax and percentage error more than 20%)
        ra, dec, parallax, para_err = gaiadr2_parallax(cuts=True)
        ra02, dec02, parallax02, para_err02 = gaiadr2_parallax(cuts=0.2)
        ra01, dec01, parallax01, para_err01 = gaiadr2_parallax(cuts=0.1)
        # assert no -9999.
        self.assertEqual(np.any([parallax == -9999.]), False)
        # assert cuts = True equals 0.2
        self.assertEqual(np.any([ra == ra02]), True)
        self.assertEqual((ra01.shape[0] < ra02.shape[0]), True)
        # assert no ridiculous parallax if do cut
        self.assertEqual(np.any([((para_err / parallax) > 0.2) & (parallax < 0.)]), False)

        ra, dec, parallax, para_err = gaiadr2_parallax(cuts=False)
        # assert some -9999. due to no cuts
        self.assertEqual(np.any([parallax == -9999.]), True)
        ra, dec, parallax, para_err = gaiadr2_parallax(cuts=True, keepdims=True)


if __name__ == '__main__':
    unittest.main()
