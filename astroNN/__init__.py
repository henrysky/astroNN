from pkg_resources import get_distribution

from .config import astroNN_CACHE_DIR
from .config import magic_num_reader, multiprocessing_flag_reader, envvar_warning_flag_reader

__version__ = get_distribution('astroNN').version

# Constant from configuration file
MAGIC_NUMBER = magic_num_reader()

# Capitalize and eval, otherwise python will treat True or False as string
MULTIPROCESS_FLAG = eval(multiprocessing_flag_reader().capitalize())
ENVVAR_WARN_FLAG = eval(envvar_warning_flag_reader().capitalize())
