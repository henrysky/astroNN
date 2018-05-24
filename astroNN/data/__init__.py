# ---------------------------------------------------------#
#   astroNN.data.__init__: tools for loading data
# ---------------------------------------------------------#
import os

import astroNN


def datapath():
    """
    Get astroNN embedded data path

    :return: full path to embedded data folder
    :rtype: str
    :History: 2018-May-16 - Written - Henry Leung (University of Toronto)
    """
    return os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN')


def data_descritpion():
    """
    Print data descritpion for astroNN embedded data

    :History: 2018-May-16 - Written - Henry Leung (University of Toronto)
    """
    items = {
        'anderson_2017_dr14_parallax.npz': 'Anderson 2017 Improved Gaia TGAS parallax from Data-Driven Stellar Model',
        'dr13_contmask.npz': 'APOGEE DR13 Continuum Mask',
        'dr14_contmask.npz': 'APOGEE DR14 Continuum Mask',
        'gaiadr2_apogeedr14_parallax.npz': 'Gaia DR2 - APOGEE DR14 matches, indices corresponds to APOGEE allstar DR14 file'}

    for item in items:
        print(item, ': ', items[item])
