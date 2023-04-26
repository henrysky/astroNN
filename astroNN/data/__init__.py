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
    :History:
    | 2018-May-16 - Written - Henry Leung (University of Toronto)
    | 2019-July-02 - Updated - Henry Leung (University of Toronto)
    """
    return os.path.join(os.path.dirname(astroNN.__path__[0]), "astroNN", "data")


def data_description():
    """
    Print data description for astroNN embedded data

    :History:
    | 2018-May-16 - Written - Henry Leung (University of Toronto)
    | 2019-July-02 - Updated - Henry Leung (University of Toronto)
    """
    items = {
        "anderson_2017_dr14_parallax.npz": "Anderson 2017 improved Gaia TGAS parallax from Data-Driven Stellar Model",
        "dr13_contmask.npz": "APOGEE DR13 continuum mask from Bovy's APOGEE tools",
        "dr14_contmask.npz": "APOGEE DR14 continuum mask from Bovy's APOGEE tools",
        "dr16_contmask.npz": "APOGEE DR16 continuum mask",
        "gaiadr2_apogeedr14_parallax.npz": "Gaia DR2 - APOGEE DR14 matches, indices corresponds "
        "to APOGEE allstar DR14 file",
        "aspcap_l31c_masks.npy": "ASPCAP l31c (DR14) elements windows mask represented by bits on the 7514px spectrum",
        "tf1_12.patch": "Patch required to make astroNN fully functional with Tensorflow 1.12.x, also used in test suite",
        "tf1_14.patch": "Patch required to make astroNN fully functional with Tensorflow 1.14.x",
    }

    for item in items:
        print(item, ": ", items[item])
