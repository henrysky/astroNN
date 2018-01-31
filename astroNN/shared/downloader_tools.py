# ---------------------------------------------------------#
#   astroNN.shared.downloader_tools: shared download tools
# ---------------------------------------------------------#

from tqdm import tqdm
import hashlib

global_block_size = 65536

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

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


def sha256_checksum(filename, block_size=global_block_size):
    """
    NAME:
        sha256_checksum
    PURPOSE:
        SHA256 checksum
    INPUT:
        filename (path)
    OUTPUT:
        sha256 checksum (string)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


def sha1_checksum(filename, block_size=global_block_size):
    """
    NAME:
        sha1_checksum
    PURPOSE:
        SHA1 checksum
    INPUT:
        filename (path)
    OUTPUT:
        sha1 checksum (string)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha1.update(block)
    return sha1.hexdigest()


def md5_checksum(filename, block_size=global_block_size):
    """
    NAME:
        md5_checksum
    PURPOSE:
        MD5 checksum
    INPUT:
        filename (path)
    OUTPUT:
        md5 checksum (string)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            md5.update(block)
    return md5.hexdigest()
