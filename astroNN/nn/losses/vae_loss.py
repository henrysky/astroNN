# -----------------------------------------------------------------------#
#   astroNN.models.losses.vae: losses function for variational autoencoder
# ----------------------------------------------------------------------#
import keras.backend as K

from astroNN.nn.losses import binary_cross_entropy


def nll(y_true, y_pred):
    """
    NAME: nll
    PURPOSE:
        Negative log likelihood
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Jan-30 - Written - Henry Leung (University of Toronto)
    """
    # astroNN binary_cross_entropy gives the mean over the last axis. we require the sum
    return K.sum(binary_cross_entropy(y_true, y_pred), axis=-1)
