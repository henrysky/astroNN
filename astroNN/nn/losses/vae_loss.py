# -----------------------------------------------------------------------#
#   astroNN.models.losses.vae: losses function for variational autoencoder
# ----------------------------------------------------------------------#
import keras.backend as K

from astroNN.nn.losses import binary_cross_entropy


def nll(y_true, y_pred):
    """
    Negative log likelihood
    Mean Squared Error is a terrible choice as a reconstruction losses
    """

    # astroNN binary_cross_entropy gives the mean over the last axis. we require the sum
    return K.sum(binary_cross_entropy(y_true, y_pred), axis=-1)
