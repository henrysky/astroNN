# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#

def h5name_check(h5name):
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')
    return  None


def foldername_modelname(folder_name=None):
    """
    NAME: foldername_modelname
    PURPOSE: convert foldername to model name
    INPUT:
        folder_name = folder name
    OUTPUT: model name
    HISTORY:
        2017-Nov-20 Henry Leung
    """
    return '/model_{}.h5'.format(folder_name[-11:])