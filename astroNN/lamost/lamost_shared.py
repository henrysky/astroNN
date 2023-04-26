# ---------------------------------------------------------#
#   astroNN.lamost.lamost_shared: shared functions for lamost
# ---------------------------------------------------------#
import os


def lamost_env():
    """
    Get LAMOST environment variable

    :return: Path to LAMOST Data
    :rtype: str
    :History: 2018-Jun-17 - Written - Henry Leung (University of Toronto)
    """
    from astroNN.config import ENVVAR_WARN_FLAG

    _LAMOST = os.getenv("LASMOT_DR5_DATA")
    if _LAMOST is None and ENVVAR_WARN_FLAG is True:
        print("WARNING! LAMOST environment variable LASMOT_DR5_DATA not set")
    return _LAMOST


def lamost_default_dr(dr=None):
    """
    Check if dr argument is provided, if none then use default

    :param dr: data release
    :type dr: Union(int, NoneType)
    :return: data release
    :rtype: int
    :History: 2018-May-13 - Written - Henry Leung (University of Toronto)
    """
    # enforce dr5 restriction
    if dr is None:
        dr = 5
        print(f"dr is not provided, using default dr={dr}")
    elif dr == 5:
        pass
    else:
        raise ValueError("Only LAMOST DR5 is supported")

    return dr
