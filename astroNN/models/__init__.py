from .Apogee_BCNN import Apogee_BCNN
from .Apogee_CNN import Apogee_CNN
from .Apogee_CVAE import Apogee_CVAE
from .Cifar10_CNN import Cifar10_CNN
from .GalaxyGAN2017 import GalaxyGAN2017
from .MNIST_BCNN import MNIST_BCNN
from .StarNet2017 import StarNet2017

from astroNN import CUSTOM_MODEL_PATH

__all__ = ['Apogee_BCNN', 'Apogee_CNN', 'Apogee_CVAE', 'StarNet2017', 'GalaxyGAN2017', 'Cifar10_CNN', 'MNIST_BCNN']


def galaxy10_cnn_setup():
    """
    NAME:
        galaxy10_cnn_setup
    PURPOSE:
        setup galaxy10_cnn from cifar10_cnn with Galaxy10 parameter
    INPUT:
    OUTPUT:
        (instance): a callable instances from Cifar10_CNN with Galaxy10 parameter
    HISTORY:
        2018-Feb-09 - Written - Henry Leung (University of Toronto)
    """
    from astroNN.datasets.galaxy10 import galaxy10cls_lookup
    galaxy10_net = Cifar10_CNN()
    galaxy10_net._model_identifier = 'Galaxy10_CNN'
    targetname = []
    for i in range(10):
        targetname.extend([galaxy10cls_lookup(i)])

    galaxy10_net.targetname = targetname
    return galaxy10_net


# Jsst an alias for Galaxy10 example
Galaxy10_CNN = galaxy10_cnn_setup()


def load_folder(folder=None):
    """
    NAME:
        load_folder
    PURPOSE:
        load astroNN model object from folder
    INPUT:
        folder (string): Name of folder, or can be None
    OUTPUT:
    HISTORY:
        2017-Dec-29 - Written - Henry Leung (University of Toronto)
    """

    import numpy as np
    import os

    currentdit = os.getcwd()

    astronn_model_obj = None

    if folder is not None and os.path.isfile(os.path.join(folder, 'astroNN_model_parameter.npz')) is True:
        parameter = np.load(os.path.join(folder, 'astroNN_model_parameter.npz'))
    elif os.path.isfile('astroNN_model_parameter.npz') is True:
        parameter = np.load('astroNN_model_parameter.npz')
    elif not os.path.exists(folder):
        raise IOError('Folder not exists: {}'.format(currentdit + '/' + folder))
    else:
        raise FileNotFoundError('Are you sure this is an astroNN generated folder?')

    id = parameter['id']
    print(id)

    if id == 'Apogee_CNN':
        astronn_model_obj = Apogee_CNN()
    elif id == 'Apogee_BCNN':
        astronn_model_obj = Apogee_BCNN()
    elif id == 'Apogee_CVAE':
        astronn_model_obj = Apogee_CVAE()
    elif id == 'Cifar10_CNN':
        astronn_model_obj = Cifar10_CNN()
    elif id == 'MNIST_BCNN':
        astronn_model_obj = MNIST_BCNN()
    elif id == 'Galaxy10_CNN':
        astronn_model_obj = Galaxy10_CNN()
    elif id == 'StarNet2017':
        astronn_model_obj = StarNet2017()
    elif id == 'GalaxyGAN2017':
        astronn_model_obj = GalaxyGAN2017()
    else:
        unknown_model_message = 'Unknown model identifier, please contact astroNN developer if you have trouble.'
        # try to load custom model from CUSTOM_MODEL_PATH
        if CUSTOM_MODEL_PATH is None:
            print("\n")
            raise TypeError(unknown_model_message)
        else:
            import sys
            from importlib import import_module
            for path in CUSTOM_MODEL_PATH:
                head, tail = os.path.split(path)
                sys.path.insert(0, head)
                try:
                    model = getattr(import_module(tail.strip('.py')), str(id))
                    astronn_model_obj = model()
                except AttributeError:
                    pass
        if astronn_model_obj is None:
            print("\n")
            raise TypeError(unknown_model_message)

    astronn_model_obj.cpu_gpu_check()

    astronn_model_obj.currentdir = currentdit
    if folder is not None:
        astronn_model_obj.fullfilepath = os.path.join(astronn_model_obj.currentdir, folder)
    else:
        astronn_model_obj.fullfilepath = astronn_model_obj.currentdir

    # Must have parameter
    astronn_model_obj.input_shape = parameter['input'].tolist()  # need to convert to list because of tensorflow.keras
    astronn_model_obj.labels_shape = parameter['labels']
    astronn_model_obj.num_hidden = parameter['hidden']
    astronn_model_obj.input_mean_norm = parameter['input_mean']
    astronn_model_obj.labels_mean_norm = parameter['labels_mean']
    astronn_model_obj.input_norm_mode = parameter['input_norm_mode']
    astronn_model_obj.labels_norm_mode = parameter['labels_norm_mode']
    astronn_model_obj.batch_size = parameter['batch_size']
    astronn_model_obj.input_std_norm = parameter['input_std']
    astronn_model_obj.labels_std_norm = parameter['labels_std']
    astronn_model_obj.targetname = parameter['targetname']
    astronn_model_obj.val_size = parameter['valsize']

    # Conditional parameter depends on neural net architecture
    try:
        astronn_model_obj.num_filters = parameter['filternum']
    except KeyError:
        pass
    try:
        # need to convert to list because of keras do not want array
        astronn_model_obj.filter_len = parameter['filterlen'].tolist()
    except KeyError:
        pass
    try:
        # need to convert to int or list because of keras do not want array
        pool_length = parameter['pool_length']
        if pool_length.shape == ():  # multi-dimensional case
            astronn_model_obj.pool_length = int(parameter['pool_length'])
        else:
            astronn_model_obj.pool_length = list(parameter['pool_length'])
    except KeyError:
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
        astronn_model_obj.l2 = parameter['l2']
    except KeyError:
        pass

    astronn_model_obj.compile()
    astronn_model_obj.keras_model.load_weights(os.path.join(astronn_model_obj.fullfilepath, 'model_weights.h5'))

    print("========================================================")
    print("Loaded astroNN model, model type: {} -> {}".format(astronn_model_obj.name, id))
    print("========================================================")
    return astronn_model_obj
