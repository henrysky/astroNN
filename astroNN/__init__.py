from pkg_resources import get_distribution
import os
from .config import magic_num_reader, multiprocessing_flag_reader
from .config import astroNN_CACHE_DIR

__version__ = get_distribution('astroNN').version

# Constant from configuration file
MAGIC_NUMBER = magic_num_reader()

# Capitalize and eval, otherwise python will treat True or False as string
MULTIPROCESS_FLAG = eval(multiprocessing_flag_reader().capitalize())