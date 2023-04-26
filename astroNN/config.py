import configparser
import os
import platform

from astroNN.shared.nn_tools import cpu_fallback, gpu_memory_manage

astroNN_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".astroNN")
_astroNN_MODEL_NAME = "model_weights.h5"  # default astroNN model filename


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
    filename = "config.ini"
    fullpath = os.path.join(astroNN_CACHE_DIR, filename)

    if os.path.isfile(fullpath):
        config = configparser.ConfigParser()
        config.sections()
        config.read(fullpath)

        # tensorflow_keras is deprecated in config on 6 Match 2019
        if any("tensorflow_keras" in d for d in config.items("Basics")):
            flag = 1  # set flag 1 to indicate it needs update

    if not os.path.isfile(fullpath) or flag == 1 or flag == 2:
        if not os.path.exists(astroNN_CACHE_DIR):
            os.makedirs(astroNN_CACHE_DIR)

        # by default initial settings
        magicnum_init = -9999
        envvar_warning_flag_init = True
        custom_model_init = "None"
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
                magicnum_init = float(config["Basics"]["MagicNumber"])
            except KeyError:
                pass
            try:
                envvar_warning_flag_init = config["Basics"][
                    "EnvironmentVariableWarning"
                ]
            except KeyError:
                pass
            try:
                custom_model_init = config["NeuralNet"]["CustomModelPath"]
            except KeyError:
                pass
            try:
                cpu_fallback_init = config["NeuralNet"]["CPUFallback"]
            except KeyError:
                pass
            try:
                gpu_memratio_init = config["NeuralNet"]["GPU_Mem_ratio"]
            except KeyError:
                pass
        elif flag == 2:
            # pass because flag==2 is resetting the file
            pass
        else:
            raise ValueError("Unknown flag, it can only either be 0 or 1!")

        os_type = platform.system()

        # Windows cannot do multiprocessing, see issue #2
        if os_type == "Windows":
            multiprocessing_flag = False
        elif os_type == "Linux" or os_type == "Darwin":
            # Deliberately set to False too as recommended by Keras
            multiprocessing_flag = False
        else:
            # other system also set to False too to be safe
            multiprocessing_flag = False

        config = configparser.ConfigParser()
        config["Basics"] = {
            "MagicNumber": magicnum_init,
            "Multiprocessing_Generator": multiprocessing_flag,
            "EnvironmentVariableWarning": envvar_warning_flag_init,
        }
        config["NeuralNet"] = {
            "CustomModelPath": custom_model_init,
            "CPUFallback": cpu_fallback_init,
            "GPU_Mem_ratio": gpu_memratio_init,
        }

        with open(fullpath, "w") as configfile:
            config.write(configfile)
            configfile.close()

        if flag == 1:
            print(
                f"astroNN just migrated the old config.ini to the new one located at {astroNN_CACHE_DIR}, "
                f"please check to make sure !!"
            )

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
        return float(config["Basics"]["MagicNumber"])
    except KeyError:
        config_path(flag=1)
        return magic_num_reader()


def multiprocessing_flag_reader():
    """
    NAME: multiprocessing_flag_readertf.keras
    PURPOSE: to read multiprocessing flag from configuration file
    INPUT:
    OUTPUT:
        (boolean)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        string = config["Basics"]["Multiprocessing_Generator"]
        return True if string.upper() == "TRUE" else False
    except KeyError:
        config_path(flag=1)
        return multiprocessing_flag_reader()


def envvar_warning_flag_reader():
    """
    NAME: envvar_warning_flag_reader
    PURPOSE: to read environment variable warning flag from configuration file
    INPUT:
    OUTPUT:
        (boolean)
    HISTORY:
        2018-Feb-10 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        string = config["Basics"]["EnvironmentVariableWarning"]
        return True if string.upper() == "TRUE" else False
    except KeyError:
        config_path(flag=1)
        return envvar_warning_flag_reader()


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
        string = config["NeuralNet"]["CustomModelPath"]
        if string.upper() != "NONE":
            string = string.split(";")
            i = 0
            while i < len(string):
                string[i] = os.path.expanduser(string[i])
                if not os.path.isfile(string[i]):
                    print(
                        f'astroNN cannot find "{string[i]}" on your system, deleted from model path reader'
                    )
                    print(
                        f'Please go and check "custommodelpath" in configuration file located at {cpath}'
                    )
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
        (boolean)
    HISTORY:
        2018-Mar-14 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.read(cpath)

    try:
        cpu_string = config["NeuralNet"]["CPUFallback"]
        gpu_string = config["NeuralNet"]["GPU_Mem_ratio"]
        cpu_string = True if cpu_string.upper() == "TRUE" else False
        gpu_string = True if gpu_string.upper() == "TRUE" else False
        return cpu_string, gpu_string
    except KeyError:
        config_path(flag=1)
        return cpu_gpu_reader()


def cpu_gpu_check():
    fallback_cpu, limit_gpu_mem = cpu_gpu_reader()
    if fallback_cpu is True:
        cpu_fallback()
    if isinstance(limit_gpu_mem, float) is True:
        gpu_memory_manage(ratio=limit_gpu_mem)
    else:
        gpu_memory_manage()


# Constant from configuration file
MAGIC_NUMBER = magic_num_reader()
MULTIPROCESS_FLAG = multiprocessing_flag_reader()
ENVVAR_WARN_FLAG = envvar_warning_flag_reader()
CUSTOM_MODEL_PATH = custom_model_path_reader()


def tf_patch():
    """
    | Tensorflow patching function
    | Usually it is just a few lines patch not merged by Tensorflow in a specific version
    """
    __tf_patches(method="patch")


def tf_unpatch():
    """
    | Tensorflow unpatching function
    | Usually it is just a few lines patch not merged by Tensorflow in a specific version
    """
    __tf_patches(method="unpatch")


def __tf_patches(method="patch"):
    """
    Internal Tensorflow patch/unpatch function

    :param method: either 'patch' or 'unpatch'
    :type method: str

    :return: None
    """
    import astroNN.data
    from astroNN.shared.patch_util import Patch
    import tensorflow as tf
    from tensorflow.python import keras
    import tensorflow.python as tfpython

    from packaging import version

    def __master_patch(path, diffpatch):
        # generate patch and patch
        patch = Patch(diffpatch)
        if method == "patch":
            action_name = "Patched"
            err = patch.apply(path)
        elif method == "unpatch":
            action_name = "Unpatched"
            err = patch.revert(path)
        else:
            raise ValueError(f"Unknown method={method}")

        if err == 0:
            print(f"{action_name} Successfully at {path}!")
        else:
            print("Error Occurred!")

    tf_ver = tf.__version__

    if version.parse("1.12.0") <= version.parse(tf_ver) < version.parse("1.13.0"):
        diff = os.path.join(astroNN.data.datapath(), "tf1_12.patch")
        patch_file_path = keras.engine.training_generator.__file__
        __master_patch(patch_file_path, diff)
    elif version.parse("1.14.0") <= version.parse(tf_ver) < version.parse("1.15.0"):
        diff = os.path.join(astroNN.data.datapath(), "tf1_14.patch")
        patch_file_path = keras.engine.network.__file__
        __master_patch(patch_file_path, diff)
    # https://github.com/tensorflow/tensorflow/pull/47957
    elif version.parse("2.5.0") <= version.parse(tf_ver) < version.parse("2.6.0"):
        diff = os.path.join(astroNN.data.datapath(), "tf2_5.patch")
        patch_file_path = tfpython.ops.array_ops.__file__
        __master_patch(patch_file_path, diff)
    else:
        print(f"Your version of Tensorflow {tf_ver} has nothing to patch")
