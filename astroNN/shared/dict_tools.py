import numpy as np


def dict_np_2_dict_list(input_dict):
    """
    Convert a dict of numpy array to a dict of list
    """
    if type(input_dict) is dict:
        for name in list(input_dict.keys()):
            input_dict.update({name: input_dict[name].tolist()})
        return input_dict
    else:
        return input_dict.tolist()


def dict_list_2_dict_np(input_dict):
    """
    Convert a dict of list to a dict of numpy array
    """
    if type(input_dict) is dict:
        for name in list(input_dict.keys()):
            input_dict.update({name: np.array(input_dict[name])})
        return input_dict
    else:
        return np.array(input_dict)
