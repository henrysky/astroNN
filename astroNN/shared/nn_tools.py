# ---------------------------------------------------------#
#   astroNN.shared.nn_tools: shared NN tools
# ---------------------------------------------------------#

def h5name_check(h5name):
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')
    return  None