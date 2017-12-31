from .bayesian_cnn import BCNN
from .cnn import CNN
from .models_shared import ModelStandard
from .starnet import StarNet
from .vae import VAE

__all__ = [BCNN, CNN, ModelStandard, StarNet, VAE]


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
    from keras.models import load_model

    currentdit = os.getcwd()

    try:
        id = np.load(os.path.join(folder, 'astroNN_use_only', 'astroNN_identifier.npy'))
    except FileNotFoundError:
        raise FileNotFoundError('Are you sure this is an astroNN generated foler?')

    if id == 'StarNet':
        astronn_model_obj = StarNet()
    elif id == 'CNN':
        astronn_model_obj = CNN()
    elif id == 'CVAE':
        astronn_model_obj = VAE()
    elif id == 'BCNN-MC':
        astronn_model_obj = BCNN()
    else:
        raise TypeError('Unknown model identifier, please contact astroNN developer if you have trouble.')

    astronn_model_obj.currentdir = currentdit
    astronn_model_obj.fullfilepath = os.path.join(astronn_model_obj.currentdir, folder)
    try:
        astronn_model_obj.target = np.load(astronn_model_obj.fullfilepath + '/targetname.npy')
    except FileNotFoundError:
        pass
    astronn_model_obj.input_shape = np.load(astronn_model_obj.fullfilepath + '/astroNN_use_only/input.npy')
    astronn_model_obj.output_shape = np.load(astronn_model_obj.fullfilepath + '/astroNN_use_only/output.npy')
    np.load(astronn_model_obj.fullfilepath + '/astroNN_use_only/hidden.npy')
    np.load(astronn_model_obj.fullfilepath + '/astroNN_use_only/filternum.npy')
    int(np.load(astronn_model_obj.fullfilepath + '/astroNN_use_only/filterlen.npy'))
    try:
        astronn_model_obj.latent_dim = int(np.load(astronn_model_obj.fullfilepath + '/astroNN_use_only/latent.npy'))
    except FileNotFoundError:
        pass
    astronn_model_obj.compile()
    astronn_model_obj.keras_model.load_weights(os.path.join(astronn_model_obj.fullfilepath, 'model_weights.h5'))


    print("=====================================")
    print("Loaded astroNN model, model type: {}".format(astronn_model_obj.name))
    print("=====================================")
    return astronn_model_obj