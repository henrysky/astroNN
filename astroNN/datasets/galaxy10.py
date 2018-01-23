# ---------------------------------------------------------#
#   astroNN.datasets.galaxy10: galaxy10
# ---------------------------------------------------------#

import numpy as np
import h5py
import os
import urllib.request
from astroNN.shared.downloader_tools import TqdmUpTo


def load_data():
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

    raise NotImplementedError("Currenly not working")

    origin = 'http://astro.utoronto.ca/~bovy/'
    filename = 'Galaxy10.h5'

    complete_url = origin + filename

    cache_dir = os.path.join(os.path.expanduser('~'), '.astroNN')
    datadir = os.path.join(os.path.expanduser(cache_dir), 'datasets')
    file_hash = '123'
    hash_algorithm = 'md5'

    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fullfilename = os.path.join(datadir, filename)

    # Check if files exists
    if not os.path.isfile(fullfilename):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=complete_url.split('/')[-1]) as t:
            urllib.request.urlretrieve(complete_url, fullfilename, reporthook=t.update_to)
            print('Downloaded Galaxy10 successfully to {}'.format(fullfilename))
    else:
        print(fullfilename + ' was found!')

    with h5py.File(fullfilename, 'r') as F:
        x = np.array(F['images'])
        y = np.array(F['ans'])

    return x, y