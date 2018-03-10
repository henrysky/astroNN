from pkg_resources import get_distribution

from .config import astroNN_CACHE_DIR
from .config import magic_num_reader, multiprocessing_flag_reader, envvar_warning_flag_reader, tf_keras_flag_reader\
    , custom_model_path_reader

__version__ = get_distribution('astroNN').version

# Constant from configuration file
MAGIC_NUMBER = magic_num_reader()
MULTIPROCESS_FLAG = multiprocessing_flag_reader()
ENVVAR_WARN_FLAG = envvar_warning_flag_reader()
TF_KERAS_FLAG = tf_keras_flag_reader()
CUSTOM_MODEL_PATH = custom_model_path_reader()


def keras_import_manager():
    """
    NAME: keras_import_manager
    PURPOSE: to import either keras or tensorflow.keras
    INPUT:
    OUTPUT:
        (string)
    HISTORY:
        2018-Mar-04 - Written - Henry Leung (University of Toronto)
    """
    if TF_KERAS_FLAG == 'AUTO':
        try:
            import keras
            return keras
        except ImportError or ModuleNotFoundError:
            try:
                import tensorflow as tf
                return tf.keras
            except ImportError or ModuleNotFoundError:
                raise ModuleNotFoundError('astroNN cannot import neither Keras nor Tensorflow')
    elif TF_KERAS_FLAG == 'TENSORFLOW':
        try:
            import tensorflow as tf
            return tf.keras
        except ImportError or ModuleNotFoundError:
            raise ModuleNotFoundError('You forced astroNN to use tensorflow.keras, but tensorflow not found')
    elif TF_KERAS_FLAG == 'KERAS':
        try:
            import keras
            return keras
        except ImportError or ModuleNotFoundError:
            raise ModuleNotFoundError('You forced astroNN to use keras, but keras not found')
    else:
        raise ValueError('Unknown option, only available option are auto, tensorflow or keras')


def switch_keras(flag=None):
    """
    NAME: switch_keras
    PURPOSE: to switch between keras or tensorflow.keras without changing the config file
    INPUT:
        flag (string): either keras or tensorflow
    OUTPUT:
        (string)
    HISTORY:
        2018-Mar-07 - Written - Henry Leung (University of Toronto)
    """
    if flag is None or (flag.upper() != 'TENSORFLOW' and flag.upper() != 'KERAS'):
        raise ValueError('flag cannot be None, it should either be tensorflow or keras')

    global TF_KERAS_FLAG
    TF_KERAS_FLAG = flag.upper()

    return None
