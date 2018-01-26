from pkg_resources import get_distribution
import os
from .config import magic_num_reader
from .config import astroNN_CACHE_DIR

__version__ = get_distribution('astroNN').version

MAGIC_NUMBER = magic_num_reader()
