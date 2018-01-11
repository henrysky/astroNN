from .APOGEE_BCNN import APOGEE_BCNN
from .APOGEE_CNN import APOGEE_CNN
from .APOGEE_CVAE import APOGEE_CVAE
from .StarNet2017 import StarNet2017
from .GalaxyGAN2017 import GalaxyGAN2017

__all__ = [APOGEE_BCNN, APOGEE_CNN, APOGEE_CVAE, StarNet2017]


def load_folder(folder):
    """
    NAME:
        load_folder
    PURPOSE:
        load astroNN model object from folder
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-29 - Written - Henry Leung (University of Toronto)
    """

    import numpy as np
    import os

    try:
        parameter = np.load(os.path.join(folder, 'astroNN_model_parameter.npz'))
    except FileNotFoundError:
        raise FileNotFoundError('Are you sure this is an astroNN generated foler?')

    id = parameter['id']

    if id == 'APOGEE_CNN':
        astronn_model_obj = APOGEE_CNN()
    elif id == 'APOGEE_CVAE':
        astronn_model_obj = APOGEE_CVAE()
    elif id == 'APOFEE_BCNN':
        astronn_model_obj = APOGEE_BCNN()
    elif id == 'StarNet2017':
        astronn_model_obj = StarNet2017()
    elif id == 'GalaxyGAN2017':
        astronn_model_obj = GalaxyGAN2017()
    else:
        print("\n")
        raise TypeError('Unknown model identifier, please contact astroNN developer if you have trouble.')

    currentdit = os.getcwd()

    astronn_model_obj.currentdir = currentdit
    astronn_model_obj.fullfilepath = os.path.join(astronn_model_obj.currentdir, folder)
    try:
        data_temp = np.load(astronn_model_obj.fullfilepath + '/targetname.npy')
        astronn_model_obj.target = data_temp
    except FileNotFoundError:
        pass
    astronn_model_obj.input_shape = parameter['input']
    astronn_model_obj.labels_shape = parameter['labels']
    astronn_model_obj.num_hidden = parameter['hidden']
    try:
        astronn_model_obj.num_filters = parameter['filternum']
    except KeyError:
        pass
    try:
        astronn_model_obj.filter_length = parameter['filterlen']
    except KeyError:
        pass
    try:
        astronn_model_obj.latent_dim = parameter['latent']
    except KeyError:
        pass
    try:
        astronn_model_obj.task = parameter['task']
    except KeyError:
        pass
    try:
        astronn_model_obj.inv_model_precision = parameter['inv_tau']
    except KeyError:
        pass
    astronn_model_obj.input_mean_norm = parameter['input_mean']
    astronn_model_obj.labels_mean_norm = parameter['labels_mean']
    astronn_model_obj.input_std_norm = parameter['input_std']
    astronn_model_obj.labels_std_norm = parameter['labels_std']

    astronn_model_obj.compile()
    astronn_model_obj.keras_model.load_weights(os.path.join(astronn_model_obj.fullfilepath, 'model_weights.h5'))

    print("==========================================")
    print("Loaded astroNN model, model type: {} -> {}".format(astronn_model_obj.name, id))
    print("==========================================")
    return astronn_model_obj
