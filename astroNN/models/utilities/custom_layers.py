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
