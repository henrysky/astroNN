# ---------------------------------------------------------#
#   astroNN.models.models_shared: Shared across models
# ---------------------------------------------------------#
from keras.models import load_model
import os


def load_from_folder_internal(modelobj, foldername):
    model = load_model(os.path.join(modelobj.currentdir, foldername, 'model.h5'))
    return model
