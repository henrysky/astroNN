# ---------------------------------------------------------#
#   astroNN.gaiatools.misc_tools: Misc. tools for Gaia detaset
# ---------------------------------------------------------#

import numpy as np


def to_absmag(mag, parallax):
    """
    NAME: to_absmag
    PURPOSE: To convert appearant magnitude to absolute magnitude
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    return mag + 5 * (np.log(parallax / 1000) + 1)
