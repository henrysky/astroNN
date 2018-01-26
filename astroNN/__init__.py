from pkg_resources import get_distribution
import os
from .config import magic_num_reader

__version__ = get_distribution('astroNN').version

astroNN_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroNN')

MAGIC_NUMBER = magic_num_reader()
