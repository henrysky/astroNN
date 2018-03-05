from pkg_resources import get_distribution

from .config import astroNN_CACHE_DIR
from .config import magic_num_reader, multiprocessing_flag_reader, envvar_warning_flag_reader, tf_keras_flag_reader

__version__ = get_distribution('astroNN').version

# Constant from configuration file
MAGIC_NUMBER = magic_num_reader()
MULTIPROCESS_FLAG = multiprocessing_flag_reader()
ENVVAR_WARN_FLAG = envvar_warning_flag_reader()
TF_KERAS_FLAG = tf_keras_flag_reader()


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
                raise ModuleNotFoundError ('astroNN cannot import neither Keras nor Tensorflow')
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
