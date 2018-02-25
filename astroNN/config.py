import configparser
import os
import platform

astroNN_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroNN')


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
        else:
            raise ValueError('Unknown flag, it can only either be 0 or 1!')

        os_type = platform.system()

        # Windows cannot do multiprocessing, see issue #2
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
            print('astroNN should has migrated the old config.ini to the new one, please check to make sure !!')
    else:
        raise ValueError('Unknown flag, it can only either be 1 or 2')

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
        string = config['Basics']['Multiprocessing_Generator']
        return True if string == 'True' else False
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
        string = config['Basics']['EnvironmentVariableWarning']
        return True if string == 'True' else False
    except KeyError:
        config_path(flag=1)
        envvar_warning_flag_reader()
        return config['Basics']['EnvironmentVariableWarning']
