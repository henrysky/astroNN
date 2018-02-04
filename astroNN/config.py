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
        if flag == 1:
            print('======Important======')
            print('astroNN just reset your astroNN config file located at {}'.format(astroNN_CACHE_DIR))

        if not os.path.exists(astroNN_CACHE_DIR):
            os.makedirs(astroNN_CACHE_DIR)

        os_type = platform.system()

        # Windows cannot do multiprocessing
        if os_type == 'Windows':
            multiprocessing_flag = False
        elif os_type == 'Linux' or os_type == 'Darwin':
            multiprocessing_flag = True
        else:
            multiprocessing_flag = False

        config = configparser.ConfigParser()
        config['Basics'] = {'MagicNumber': '-9999.', 'Multiprocessing_Generator': multiprocessing_flag}

        with open(fullpath, 'w') as configfile:
            config.write(configfile)

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
        (float)
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
