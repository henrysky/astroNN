from .chips import wavelength_solution, pseudo_continuum
from .lamost_shared import lamost_default_dr, lamost_env


def load_allstar_dr5():
    """
    Open LAMOST DR5 allstar

    :return: fits file opened by astropy
    :rtype: astropy.io.fits.hdu.hdulist.HDUList
    :History: 2018-Jun-17 - Written - Henry Leung (University of Toronto)
    """
    import os
    from astropy.io import fits

    file_name = "LAMO5_2MS_AP9_SD14_UC4_PS1_AW_Carlin_M.fits"
    _lamost_dr5_allsta_path = os.path.join(lamost_env(), "DR5", file_name)
    if not os.path.isfile(_lamost_dr5_allsta_path):
        raise FileNotFoundError(f"{file_name} file not found")
    return fits.open(_lamost_dr5_allsta_path)
