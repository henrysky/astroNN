# ---------------------------------------------------------#
# Utilities to handle dictionary
# ---------------------------------------------------------#
import copy
import numpy as np


def dict_np_to_dict_list(input_dict):
    """
    Convert a dict of numpy array to a dict of list
    """
    input_dict = copy.copy(input_dict)
    if type(input_dict) is dict:
        for name in list(input_dict.keys()):
            input_dict.update({name: input_dict[name].tolist()})
        return input_dict
    else:
        return input_dict.tolist()


def dict_list_to_dict_np(input_dict):
    """
    Convert a dict of list to a dict of numpy array
    """
    input_dict = copy.copy(input_dict)
    if type(input_dict) is dict:
        for name in list(input_dict.keys()):
            input_dict.update({name: np.array(input_dict[name])})
        return input_dict
    else:
        return np.array(input_dict)


def list_to_dict(names, arrs):
    """
    Matching a list of array with names
    """
    # TODO: need detailed test
    if type(arrs) is list:
        final_dict = {}
        if len(names) == len(arrs):
            for name, arr in zip(names, arrs):
                final_dict.update({name: arr})
        elif len(arrs):
            for name in names:
                final_dict.update({name: arrs[0]})
        else:
            raise IndexError(
                f"names has a length of {len(names)} but arrs has a  length of {len(arrs)}"
            )
        return final_dict
    elif type(arrs) is np.ndarray and len(names) == 1:
        return {names[0]: arrs}
    elif type(arrs) is np.ndarray and len(names) > 1:
        final_dict = {}
        for name in names:
            final_dict.update({name: arrs})
        return final_dict
    else:
        return arrs


def to_iterable(var):
    """
    convert things to list
    treat string as not iterable!
    """
    try:
        if type(var) is str:
            raise Exception
        iter(var)
    except Exception:
        return [var]
    else:
        return var
