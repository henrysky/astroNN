import configparser
import os
import platform

astroNN_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroNN')


def config_path(flag=None):
    """
    NAME: config_path
    PURPOSE: get configuration file path
    INPUT:
        flag (boolean): 1 or True to reset the config file
    OUTPUT:
        (path)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    filename = 'config.ini'
    fullpath = os.path.join(astroNN_CACHE_DIR, filename)

    if not os.path.isfile(fullpath) or flag == 1:
        if not os.path.exists(astroNN_CACHE_DIR):
            os.makedirs(astroNN_CACHE_DIR)

        magicnum_init = -9999
        envvar_warning_flag_init = True

        # Set flag back to 0 as flag=1 probably just because the file not even exists (example: first time using it)
        if not os.path.isfile(fullpath):
            flag = 0
        else:
            config = configparser.ConfigParser()
            config.sections()
            config.read(fullpath)
            try:
                magicnum_init = float(config['Basics']['MagicNumber'])
            except KeyError:
                pass
            try:
                envvar_warning_flag_init = config['Basics']['EnvironmentVariableWarning']
            except KeyError:
                pass

        os_type = platform.system()

        # Windows cannot do multiprocessing
        if os_type == 'Windows':
            multiprocessing_flag = False
        elif os_type == 'Linux' or os_type == 'Darwin':
            multiprocessing_flag = True
        else:
            multiprocessing_flag = False

        config = configparser.ConfigParser()
        config['Basics'] = {'MagicNumber': magicnum_init,
                            'Multiprocessing_Generator': multiprocessing_flag,
                            'EnvironmentVariableWarning': envvar_warning_flag_init}

        with open(fullpath, 'w') as configfile:
            config.write(configfile)

        if flag == 1:
            print('=================Important=================')
            print('astroNN just updated your astroNN config file located at {}'.format(astroNN_CACHE_DIR))
            print('astroNN should migrated the old config.ini to the new one but please check to make sure !!')

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
    config.sections()
    config.read(cpath)

    try:
        return float(config['Basics']['MagicNumber'])
    except KeyError:
        config_path(flag=1)
        magic_num_reader()


def multiprocessing_flag_reader():
    """
    NAME: multiprocessing_flag_reader
    PURPOSE: to read multiprocessing flag from configuration file
    INPUT:
    OUTPUT:
        (string or boolean)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    cpath = config_path()
    config = configparser.ConfigParser()
    config.sections()
    config.read(cpath)

    try:
        return config['Basics']['Multiprocessing_Generator']
    except KeyError:
        config_path(flag=1)
        multiprocessing_flag_reader()
        return config['Basics']['Multiprocessing_Generator']


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
    config.sections()
    config.read(cpath)

    try:
        return config['Basics']['EnvironmentVariableWarning']
    except KeyError:
        config_path(flag=1)
        envvar_warning_flag_reader()
        return config['Basics']['EnvironmentVariableWarning']
