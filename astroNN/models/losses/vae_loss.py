# -----------------------------------------------------------------------#
#   astroNN.models.losses.vae: losses function for variational autoencoder
# ----------------------------------------------------------------------#
import keras.backend as K
from keras import metrics


def nll(y_true, y_pred):
    """
    Negative log likelihood
    Mean Squared Error is a terrible choice as a reconstruction losses
    """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)