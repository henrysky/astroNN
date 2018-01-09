# -----------------------------------------------------------------------#
#   astroNN.models.loss.vae: loss function for variational autoencoder
# ----------------------------------------------------------------------#
import keras.backend as K
from keras import metrics


def vae_loss(x, x_decoded_mean, z_mean, z_log_var):
    shape = int(x.shape[1])
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = shape * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)