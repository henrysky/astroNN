# ---------------------------------------------------------#
#   astroNN.datasets.galaxy10: galaxy10
# ---------------------------------------------------------#

import os
import urllib.request

import h5py
import numpy as np

from astroNN.config import astroNN_CACHE_DIR
from astroNN.shared.downloader_tools import TqdmUpTo, filehash

Galaxy10Class = {
    0: "Disk, Face-on, No Spiral",
    1: "Smooth, Completely round",
    2: "Smooth, in-between round",
    3: "Smooth, Cigar shaped",
    4: "Disk, Edge-on, Rounded Bulge",
    5: "Disk, Edge-on, Boxy Bulge",
    6: "Disk, Edge-on, No Bulge",
    7: "Disk, Face-on, Tight Spiral",
    8: "Disk, Face-on, Medium Spiral",
    9: "Disk, Face-on, Loose Spiral",
}

_G10_ORIGIN = "https://www.astro.utoronto.ca/~bovy/Galaxy10/"


def load_data(flag=None):
    """
    NAME:
        load_data
    PURPOSE:
        load_data galaxy10 data
    INPUT:
        None
    OUTPUT:
        x (ndarray): An array of images
        y (ndarray): An array of answer
    HISTORY:
        2018-Jan-22 - Written - Henry Leung (University of Toronto)
    """

    filename = "Galaxy10.h5"

    complete_url = _G10_ORIGIN + filename

    datadir = os.path.join(astroNN_CACHE_DIR, "datasets")
    file_hash = (
        "969A6B1CEFCC36E09FFFA86FEBD2F699A4AA19B837BA0427F01B0BC6DED458AF"  # SHA256
    )

    # Notice python expect sha256 in lowercase

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fullfilename = os.path.join(datadir, filename)

    # Check if files exists
    if os.path.isfile(fullfilename) and flag is None:
        checksum = filehash(fullfilename, algorithm="sha256")
        if checksum != file_hash.lower():
            print("File corruption detected, astroNN is attempting to download again")
            load_data(flag=1)
        else:
            print(fullfilename + " was found!")
    elif not os.path.isfile(fullfilename) or flag == 1:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=complete_url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                complete_url, fullfilename, reporthook=t.update_to
            )
            print(f"Downloaded Galaxy10 successfully to {fullfilename}")
            checksum = filehash(fullfilename, algorithm="sha256")
            if checksum != file_hash.lower():
                load_data(flag=1)

    with h5py.File(fullfilename, "r") as F:
        x = np.array(F["images"])
        y = np.array(F["ans"])

    return x, y


def galaxy10cls_lookup(class_num):
    """
    NAME:
        galaxy10cls_lookup
    PURPOSE:
        look up class name for Galaxy10
    INPUT:
        class_num (int): An integer 0-9
    OUTPUT:
        (string): Name of the class
    HISTORY:
        2018-Feb-07 - Written - Henry Leung (University of Toronto)
    """
    if isinstance(class_num, list) or isinstance(class_num, np.ndarray):
        class_num = np.argmax(class_num)
    if 0 > class_num or 9 < class_num:
        raise ValueError(
            f"Galaxy10 only has 10 classes (class 0 to class 9), you entered class {class_num}"
        )
    return Galaxy10Class[class_num]


def galaxy10_confusion(confusion_mat):
    """
    NAME:
        galaxy10_confusion
    PURPOSE:
        to plot confusion matrix
    INPUT:
        confusion_mat (ndarray): An integer 0-9
    OUTPUT:
        (string): Name of the class
    HISTORY:
        2018-Feb-11 - Written - Henry Leung (University of Toronto)
    """
    import pylab as plt

    conf_arr = confusion_mat.astype(int)

    norm_conf = []
    a = np.max(conf_arr)
    for i in conf_arr:
        tmp_arr = []
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(10, 10.5))
    ax = fig.gca()
    ax.set_title("Confusion Matrix for Galaxy10", fontsize=20)
    ax.set_aspect(1)
    ax.imshow(np.array(norm_conf), cmap=plt.get_cmap("Blues"), interpolation="nearest")

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(
                str(conf_arr[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.set_xticks(np.arange(width))
    ax.set_xticklabels([str(i) for i in range(width)], fontsize=20)
    ax.set_yticks(np.arange(height))
    ax.set_yticklabels([str(i) for i in range(width)], fontsize=20)
    ax.set_ylabel("Prediction", fontsize=20)
    ax.set_xlabel("Truth", fontsize=20)
    fig.tight_layout()

    return None
