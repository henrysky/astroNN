import configparser
import os
import platform
import warnings

from astroNN.shared.nn_tools import cpu_fallback, gpu_memory_manage

astroNN_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroNN')
_astroNN_MODEL_NAME = 'model_weights.h5'  # default astroNN model filename


def config_path(flag=None):
    """
    NAME: config_path
    PURPOSE: get configuration file path
    INPUT:
        flag (boolean): 1 to update the config file, 2 to reset the config file
    OUTPUT:
        (path)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    filename = 'config.ini'
    fullpath = os.path.join(astroNN_CACHE_DIR, filename)

    if not os.path.isfile(fullpath) or flag == 1 or flag == 2:
        if not os.path.exists(astroNN_CACHE_DIR):
            os.makedirs(astroNN_CACHE_DIR)

        magicnum_init = -9999
        envvar_warning_flag_init = True
        tf_keras_flag_init = 'auto'
        custom_model_init = 'None'
        cpu_fallback_init = False
        gpu_memratio_init = True

        # Set flag back to 0 as flag=1 probably just because the file not even exists (example: first time using it)
        if not os.path.isfile(fullpath):
            flag = 0
        # only try to  migrate the old setting to new one if flag is 1 as flag=2 for reset
        elif flag == 1:  # Try to migrate the old setting to new one
            config = configparser.ConfigParser()
            config.sections()
            config.read(fullpath)
            # Try to migrate the old setting to new one
            try:
                magicnum_init = float(config['Basics']['MagicNumber'])
            except KeyError:
                pass
            try:
                envvar_warning_flag_init = config['Basics']['EnvironmentVariableWarning']
            except KeyError:
                pass
            try:
                tf_keras_flag_init = config['Basics']['Tensorflow_Keras']
            except KeyError:
                pass
            try:
                custom_model_init = config['NeuralNet']['CustomModelPath']
            except KeyError:
                pass
            try:
                cpu_fallback_init = config['NeuralNet']['CPUFallback']
            except KeyError:
                pass
            try:
                gpu_memratio_init = config['NeuralNet']['GPU_Mem_ratio']
            except KeyError:
                pass
        elif flag == 2:
            # pass because flag==2 is resetting the file
            pass
        else:
            raise ValueError('Unknown flag, it can only either be 0 or 1!')

        os_type = platform.system()

        # Windows cannot do multiprocessing, see issue #2
        if os_type == 'Windows':
            multiprocessing_flag = False
        elif os_type == 'Linux' or os_type == 'Darwin':
            # Delibratly set to False too as recommended by Keras
            multiprocessing_flag = False
        else:
            multiprocessing_flag = False

        config = configparser.ConfigParser()
        config['Basics'] = {'MagicNumber': magicnum_init,
                            'Multiprocessing_Generator': multiprocessing_flag,
                            'EnvironmentVariableWarning': envvar_warning_flag_init,
                            'Tensorflow_Keras': tf_keras_flag_init}
        config['NeuralNet'] = {'CustomModelPath': custom_model_init,
                               'CPUFallback': cpu_fallback_init,
                               'GPU_Mem_ratio': gpu_memratio_init}

        with open(fullpath, 'w') as configfile:
            config.write(configfile)
            configfile.close()

        if flag == 1:
            print('=================Important=================')
            print(f'astroNN just migrated the old config.ini to the new one located at {astroNN_CACHE_DIR}, '
                  f'please check to make sure !!')

    return fullpath


def magic_num_reader():
    """
    NAME: magic_num_reader
    PURPOSE: to read magic number from configuration file
    INPUT:
    OUTPUT:
        (float)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        return float(config['Basics']['MagicNumber'])
    except KeyError:
        config_path(flag=1)
        return magic_num_reader()


def multiprocessing_flag_reader():
    """
    NAME: multiprocessing_flag_readertf.keras
    PURPOSE: to read multiprocessing flag from configuration file
    INPUT:
    OUTPUT:
        (string or boolean)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        string = config['Basics']['Multiprocessing_Generator']
        return True if string.upper() == 'TRUE' else False
    except KeyError:
        config_path(flag=1)
        return multiprocessing_flag_reader()


def envvar_warning_flag_reader():
    """
    NAME: envvar_warning_flag_reader
    PURPOSE: to read environment variable warning flag from configuration file
    INPUT:
    OUTPUT:
        (string or boolean)
    HISTORY:
        2018-Feb-10 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        string = config['Basics']['EnvironmentVariableWarning']
        return True if string.upper() == 'TRUE' else False
    except KeyError:
        config_path(flag=1)
        return envvar_warning_flag_reader()


def tf_keras_flag_reader():
    """
    NAME: tf_keras_flag_reader
    PURPOSE: to read flag to decide whether using keras or tensorflow.keras or auto decided by astroNN
    INPUT:
    OUTPUT:
        (string)
    HISTORY:
        2018-Feb-10 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        string = config['Basics']['Tensorflow_Keras']
        return string.upper()
    except KeyError:
        config_path(flag=1)
        return tf_keras_flag_reader()


def custom_model_path_reader():
    """
    NAME: custom_model_path_reader
    PURPOSE: to read path of custom models
    INPUT:
    OUTPUT:
        (string)
    HISTORY:
        2018-Mar-09 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        string = config['NeuralNet']['CustomModelPath']
        if string.upper() != 'NONE':
            string = string.split(';')
            i = 0
            while i < len(string):
                string[i] = os.path.expanduser(string[i])
                if not os.path.isfile(string[i]):
                    print(f'astroNN cannot find "{string[i]}" on your system, deleted from model path reader')
                    print(f'Please go and check "custommodelpath" in configuration file located at {cpath}')
                    del string[i]
                else:
                    i += 1
            return string
        else:
            return None
    except KeyError:
        config_path(flag=1)
        return custom_model_path_reader()


def cpu_gpu_reader():
    """
    NAME: cpu_gpu_reader
    PURPOSE: to read cpu gpu setting in config
    INPUT:
    OUTPUT:
        (string)
    HISTORY:
        2018-Mar-14 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        cpu_string = config['NeuralNet']['CPUFallback']
        gpu_string = config['NeuralNet']['GPU_Mem_ratio']
        cpu_string = True if cpu_string.upper() == 'TRUE' else False
        gpu_string = True if gpu_string.upper() == 'TRUE' else False
        return cpu_string, gpu_string
    except KeyError:
        config_path(flag=1)
        return cpu_gpu_reader()


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
    def try_tf():
        import tensorflow as tf
        warnings.warn('Please be warned that tensorflow.keras is not fully supported, please install keras '
                      'separately if you encountered any issue')
        return tf.keras

    if TF_KERAS_FLAG == 'AUTO':
        try:
            import keras
            return keras
        except ImportError or ModuleNotFoundError:
            try:
                return try_tf()
            except ImportError or ModuleNotFoundError:
                raise ModuleNotFoundError('astroNN cannot import neither Keras nor Tensorflow')
    elif TF_KERAS_FLAG == 'TENSORFLOW':
        try:
            return try_tf()
        except ImportError or ModuleNotFoundError:
            raise ModuleNotFoundError('Tensorflow not found! You must install Tensorflow to use astroNN')
    elif TF_KERAS_FLAG == 'KERAS':
        try:
            import keras
            return keras
        except ImportError or ModuleNotFoundError:
            raise ModuleNotFoundError('You have forced astroNN to use keras, but keras not found!')
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


def cpu_gpu_check():
    fallback_cpu, limit_gpu_mem = cpu_gpu_reader()
    if fallback_cpu is True:
        cpu_fallback()
    if limit_gpu_mem is True:
        gpu_memory_manage()
    elif isinstance(limit_gpu_mem, float) is True:
        gpu_memory_manage(ratio=limit_gpu_mem)


# Constant from configuration file
MAGIC_NUMBER = magic_num_reader()
MULTIPROCESS_FLAG = multiprocessing_flag_reader()
ENVVAR_WARN_FLAG = envvar_warning_flag_reader()
TF_KERAS_FLAG = tf_keras_flag_reader()
CUSTOM_MODEL_PATH = custom_model_path_reader()
