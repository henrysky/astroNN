# ---------------------------------------------------------#
#   astroNN.shared.downloader_tools: shared download tools
# ---------------------------------------------------------#

import hashlib

from tqdm import tqdm


class TqdmUpTo(tqdm):
    """
    NAME:
        sha256_checksum
    PURPOSE:
        Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Oct-25 - Written - Henry Leung (University of Toronto)
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def filehash(filename, block_size=65536, algorithm="sha256"):
    """
    Computes the hash value for a file by using a specified hash algorithm.

    :param filename: filename
    :type filename: str
    :param block_size: blocksize used to compute file hash
    :type block_size: int
    :param algorithm: hash algorithms like 'sha256' or 'md5' etc.
    :type algorithm: str
    :return: None
    :History: 2019-Mar-12 - Written - Henry Leung (University of Toronto)
    """
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_guaranteed:
        raise ValueError(f"{algorithm} is an unsupported hashing algorithm")

    func_algorithm = getattr(hashlib, algorithm)()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            func_algorithm.update(block)
    return func_algorithm.hexdigest()
