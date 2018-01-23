from pkg_resources import get_distribution
import os

__version__ = get_distribution('astroNN').version

astroNN_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroNN')

MAGIC_NUMBER = -9999.
