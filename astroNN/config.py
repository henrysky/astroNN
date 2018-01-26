import configparser
import os

astroNN_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.astroNN')


def config_path():
    """
    NAME: config_path
    PURPOSE: get configuration file path
    INPUT:
    OUTPUT:
        (path)
    HISTORY:
        2018-Jan-25 - Written - Henry Leung (University of Toronto)
    """
    filename = 'config.ini'
    fullpath = os.path.join(astroNN_CACHE_DIR, filename)

    if not os.path.isfile(fullpath):
        if not os.path.exists(astroNN_CACHE_DIR):
            os.makedirs(astroNN_CACHE_DIR)

        config = configparser.ConfigParser()
        config['Basics'] = {'MagicNumber': '-9999.'}

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

    return config['Basics']['MagicNumber']
