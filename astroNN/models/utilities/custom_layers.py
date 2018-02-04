import keras.backend as K
from keras.layers import Layer


class KLDivergenceLayer(Layer):
    """
    Identity transform layer that adds KL divergence to the final model losses.
    KL divergence used to force the latent space match the prior (in this case its unit gaussian)
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class TimeDistributedMeanVar(Layer):
    """
    NAME: TimeDistributedMeanVar
    PURPOSE: Take mean and variance of the results of a TimeDistributed layer.
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-02 - Written - Henry Leung (University of Toronto)
    """

    def build(self, input_shape):
        super(TimeDistributedMeanVar, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]

    def call(self, x):
        return K.mean(x, axis=1), K.var(x, axis=1)
