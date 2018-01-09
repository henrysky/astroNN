from keras.layers import Layer
from astroNN.models.loss.vae_loss import vae_loss


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    @staticmethod
    def vae_loss_warapper(x, x_decoded_mean, z_mean, z_log_var):
        return vae_loss(x, x_decoded_mean, z_mean, z_log_var)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss_warapper(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x