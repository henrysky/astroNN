import json
import os

import h5py
import numpy as np
from tensorflow import Graph, Session

from astroNN.config import keras_import_manager, custom_model_path_reader
from astroNN.models.ApogeeBCNN import ApogeeBCNN
from astroNN.models.ApogeeBCNNCensored import ApogeeBCNNCensored
from astroNN.models.ApogeeCNN import ApogeeCNN
from astroNN.models.ApogeeCVAE import ApogeeCVAE
from astroNN.models.Cifar10CNN import Cifar10CNN
from astroNN.models.MNIST_BCNN import MNIST_BCNN
from astroNN.models.SimplePolyNN import SimplePolyNN
from astroNN.models.StarNet2017 import StarNet2017
from astroNN.nn.losses import losses_lookup
from astroNN.nn.utilities import Normalizer

keras = keras_import_manager()
optimizers = keras.optimizers
Sequential = keras.models.Sequential

_GRAPH_COUTNER = 0  # keep track of the indices used in list storage below
_GRAPH_STORAGE = []  # store all the graph used by multiple models
_SESSION_STORAGE = []  # store all the graph used by multiple models


def Galaxy10CNN():
    """
    NAME:
        Galaxy10CNN
    PURPOSE:
        setup Galaxy10CNN from Cifar10CNN with Galaxy10 parameter
    INPUT:
    OUTPUT:
        (instance): a callable instances from Cifar10_CNN with Galaxy10 parameter
    HISTORY:
        2018-Feb-09 - Written - Henry Leung (University of Toronto)
        2018-Apr-02 - Update - Henry Leung (University of Toronto)
    """
    from astroNN.datasets.galaxy10 import galaxy10cls_lookup
    galaxy10_net = Cifar10CNN()
    galaxy10_net._model_identifier = 'Galaxy10CNN'
    targetname = []
    for i in range(10):
        targetname.extend([galaxy10cls_lookup(i)])

    galaxy10_net.targetname = targetname
    return galaxy10_net


__all__ = ['ApogeeBCNN', 'ApogeeBCNNCensored', 'ApogeeCNN', 'ApogeeCVAE', 'StarNet2017', 'Cifar10CNN', 'MNIST_BCNN',
           'Galaxy10CNN']


def convert_custom_objects(obj):
    """Handles custom object lookup.

    Based on Keras Source Code

    # Arguments
        obj: object, dict, or list.

    # Returns
        The same structure, where occurrences
            of a custom object name have been replaced
            with the custom object.
    """
    if isinstance(obj, list):
        deserialized = []
        for value in obj:
            deserialized.append(convert_custom_objects(value))
        return deserialized
    if isinstance(obj, dict):
        deserialized = {}
        for key, value in obj.items():
            deserialized[key] = convert_custom_objects(value)
        return deserialized
    return obj


def load_folder(folder=None):
    """
    To load astroNN model object from folder

    :param folder: [optional] you should provide folder name if outside folder, do not specific when you are inside the folder
    :type folder: str
    :return: astroNN Neural Network instance
    :rtype: astroNN.nn.NeuralNetMaster.NeuralNetMaster
    :History: 2017-Dec-29 - Written - Henry Leung (University of Toronto)
    """
    currentdir = os.getcwd()

    if folder is not None:
        fullfilepath = os.path.join(currentdir, folder)
    else:
        fullfilepath = currentdir

    astronn_model_obj = None

    if folder is not None and os.path.isfile(os.path.join(folder, 'astroNN_model_parameter.json')) is True:
        with open(os.path.join(folder, 'astroNN_model_parameter.json')) as f:
            parameter = json.load(f)
            f.close()
    elif os.path.isfile('astroNN_model_parameter.json') is True:
        with open('astroNN_model_parameter.json') as f:
            parameter = json.load(f)
            f.close()
    elif folder is not None and not os.path.exists(folder):
        raise IOError('Folder not exists: ' + str(currentdir + '/' + folder))
    else:
        raise FileNotFoundError('Are you sure this is an astroNN generated folder? Or is it a folder trained by old '
                                'astroNN version?')

    identifier = parameter['id']

    if identifier == 'ApogeeCNN':
        astronn_model_obj = ApogeeCNN()
    elif identifier == 'ApogeeBCNN':
        astronn_model_obj = ApogeeBCNN()
    elif identifier == 'ApogeeBCNNCensored':
        astronn_model_obj = ApogeeBCNNCensored()
    elif identifier == 'ApogeeCVAE':
        astronn_model_obj = ApogeeCVAE()
    elif identifier == 'Cifar10CNN':
        astronn_model_obj = Cifar10CNN()
    elif identifier == 'MNIST_BCNN':
        astronn_model_obj = MNIST_BCNN()
    elif identifier == 'Galaxy10CNN':
        astronn_model_obj = Galaxy10CNN()
    elif identifier == 'StarNet2017':
        astronn_model_obj = StarNet2017()
    elif identifier == 'SimplePolyNN':
        astronn_model_obj = SimplePolyNN()
    else:
        unknown_model_message = f'Unknown model identifier -> {identifier}!'
        # try to load custom model from CUSTOM_MODEL_PATH
        CUSTOM_MODEL_PATH = custom_model_path_reader()
        # try the current folder and see if there is any .py on top of CUSTOM_MODEL_PATH
        list_py_files = [os.path.join(fullfilepath, f) for f in os.listdir(fullfilepath) if f.endswith(".py")]
        if CUSTOM_MODEL_PATH is None and list_py_files is None:
            print("\n")
            raise TypeError(unknown_model_message)
        else:
            import sys
            from importlib import import_module
            for path_list in (path_list for path_list in [CUSTOM_MODEL_PATH, list_py_files] if path_list is not None):
                for path in path_list:
                    head, tail = os.path.split(path)
                    sys.path.insert(0, head)
                    try:
                        model = getattr(import_module(tail.strip('.py')), str(identifier))
                        astronn_model_obj = model()
                    except AttributeError:
                        pass

        if astronn_model_obj is None:
            print("\n")
            raise TypeError(unknown_model_message)

    astronn_model_obj.currentdir = currentdir
    astronn_model_obj.fullfilepath = fullfilepath
    astronn_model_obj.folder_name = folder if folder is not None else os.path.basename(os.path.normpath(currentdir))

    # Must have parameter
    astronn_model_obj._input_shape = parameter['input']
    astronn_model_obj._labels_shape = parameter['labels']
    astronn_model_obj.num_hidden = parameter['hidden']
    astronn_model_obj.input_norm_mode = parameter['input_norm_mode']
    astronn_model_obj.labels_norm_mode = parameter['labels_norm_mode']
    astronn_model_obj.input_mean = np.array(parameter['input_mean'])
    astronn_model_obj.labels_mean = np.array(parameter['labels_mean'])
    astronn_model_obj.input_std = np.array(parameter['input_std'])
    astronn_model_obj.labels_std = np.array(parameter['labels_std'])
    astronn_model_obj.batch_size = parameter['batch_size']
    astronn_model_obj.targetname = parameter['targetname']
    astronn_model_obj.val_size = parameter['valsize']

    # create normalizer and set correct mean and std
    astronn_model_obj.input_normalizer = Normalizer(mode=astronn_model_obj.input_norm_mode)
    astronn_model_obj.labels_normalizer = Normalizer(mode=astronn_model_obj.labels_norm_mode)
    astronn_model_obj.input_normalizer.mean_labels = astronn_model_obj.input_mean
    astronn_model_obj.input_normalizer.std_labels = astronn_model_obj.input_std
    astronn_model_obj.labels_normalizer.mean_labels = astronn_model_obj.labels_mean
    astronn_model_obj.labels_normalizer.std_labels = astronn_model_obj.labels_std

    # Conditional parameter depends on neural net architecture
    try:
        astronn_model_obj.num_filters = parameter['filternum']
    except KeyError:
        pass
    try:
        astronn_model_obj.filter_len = parameter['filterlen']
    except KeyError:
        pass
    try:
        pool_length = parameter['pool_length']
        if pool_length is not None:
            if isinstance(pool_length, int):  # multi-dimensional case
                astronn_model_obj.pool_length = parameter['pool_length']
            else:
                astronn_model_obj.pool_length = list(parameter['pool_length'])
    except KeyError or TypeError:
        pass
    try:
        # need to convert to int because of keras do not want array or list
        astronn_model_obj.latent_dim = int(parameter['latent'])
    except KeyError:
        pass
    try:
        astronn_model_obj.task = parameter['task']
    except KeyError:
        pass
    try:
        astronn_model_obj.dropout_rate = parameter['dropout_rate']
    except KeyError:
        pass
    try:
        # if inverse model precision exists, so does length_scale
        astronn_model_obj.inv_model_precision = parameter['inv_tau']
        astronn_model_obj.length_scale = parameter['length_scale']
    except KeyError:
        pass
    try:
        astronn_model_obj.l1 = parameter['l1']
    except KeyError:
        pass
    try:
        astronn_model_obj.l2 = parameter['l2']
    except KeyError:
        pass
    try:
        astronn_model_obj.maxnorm = parameter['maxnorm']
    except KeyError:
        pass
    try:
        astronn_model_obj._last_layer_activation = parameter['last_layer_activation']
    except KeyError:
        pass

    global _GRAPH_COUTNER
    global _GRAPH_STORAGE
    global _SESSION_STORAGE
    _GRAPH_COUTNER += 1
    _GRAPH_STORAGE.append(Graph())

    with _GRAPH_STORAGE[_GRAPH_COUTNER - 1].as_default():
        _SESSION_STORAGE.append(Session())
        with _SESSION_STORAGE[_GRAPH_COUTNER - 1].as_default():
            with h5py.File(os.path.join(astronn_model_obj.fullfilepath, 'model_weights.h5'), mode='r') as f:
                training_config = f.attrs.get('training_config')
                training_config = json.loads(training_config.decode('utf-8'))
                optimizer_config = training_config['optimizer_config']
                optimizer = optimizers.deserialize(optimizer_config)

                # Recover loss functions and metrics.
                losses = convert_custom_objects(training_config['loss'])
                try:
                    try:
                        [losses_lookup(losses[loss]) for loss in losses]
                    except TypeError:
                        losses_lookup(losses)
                except:
                    pass

                metrics = convert_custom_objects(training_config['metrics'])
                # its weird that keras needs -> metrics[metric][0] instead of metrics[metric] likes losses, need attention
                try:
                    try:
                        [losses_lookup(metrics[metric][0]) for metric in metrics]
                    except TypeError:
                        losses_lookup(metrics[0])
                except:
                    pass

                sample_weight_mode = training_config['sample_weight_mode']
                loss_weights = training_config['loss_weights']

                # compile the model
                astronn_model_obj.compile(optimizer=optimizer)

                # set weights
                astronn_model_obj.keras_model.load_weights(
                    os.path.join(astronn_model_obj.fullfilepath, 'model_weights.h5'))

                # Build train function (to get weight updates), need to consider Sequential model too
                astronn_model_obj.keras_model._make_train_function()
                if isinstance(astronn_model_obj.keras_model, Sequential):
                    astronn_model_obj.keras_model.model._make_train_function()
                else:
                    astronn_model_obj.keras_model._make_train_function()
                optimizer_weights_group = f['optimizer_weights']
                optimizer_weight_names = [n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']]
                optimizer_weight_values = [optimizer_weights_group[n] for n in optimizer_weight_names]
                astronn_model_obj.keras_model.optimizer.set_weights(optimizer_weight_values)

    astronn_model_obj.graph = _GRAPH_STORAGE[_GRAPH_COUTNER - 1]  # the graph associated with the model
    astronn_model_obj.session = _SESSION_STORAGE[_GRAPH_COUTNER - 1]  # the model associated with the model
    astronn_model_obj.session.__enter__()  # register the latest model loaded to defualt tensorflow session

    print("========================================================")
    print(f"Loaded astroNN model, model type: {astronn_model_obj.name} -> {identifier}")
    print("========================================================")
    return astronn_model_obj
