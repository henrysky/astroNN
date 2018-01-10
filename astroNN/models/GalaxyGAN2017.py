# ---------------------------------------------------------#
#   astroNN.models.GalaxyGAM2017: Contain GalaxyGAN2017 model
# ---------------------------------------------------------#
from keras.layers import MaxPooling1D, Conv2D, Dense, Flatten, LeakyReLU
from keras.models import Model, Input

from astroNN.models.NeuralNetBases import CGANBase


class GalaxyGAN2017(CGANBase):
    """
    NAME:
        GalaxyGAN2017
    PURPOSE:
        To create Convolutional Generative Adversarial Network
    HISTORY:
        2018-Jan_10 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        """
        NAME:
            model
        PURPOSE:
            To create Convolutional Generative Adversarial Network
        INPUT:
        OUTPUT:
        HISTORY:
            2018-Jan-10 - Written - Henry Leung (University of Toronto)
        """
        super(GalaxyGAN2017, self).__init__()

    class Config:
        data_path = "figures"
        model_path_train = ""
        model_path_test = "figures/checkpoint/model_20.ckpt"
        output_path = "results"

        img_size = 424
        adjust_size = 424
        train_size = 424
        img_channel = 3
        conv_channel_base = 64

        learning_rate = 0.0002
        beta1 = 0.5
        max_epoch = 20
        L1_lambda = 100
        save_per_epoch = 1

    def asda(self, img, cond, reuse):
        dim = len(img.get_shape())
        with tf.variable_scope("disc", reuse=reuse):
            image = tf.concat([img, cond], dim - 1)
            feature = conf.conv_channel_base
            h0 = lrelu(conv2d(image, feature, name="h0"))
            h1 = lrelu(batch_norm(conv2d(h0, feature*2, name="h1"), "h1"))
            h2 = lrelu(batch_norm(conv2d(h1, feature*4, name="h2"), "h2"))
            h3 = lrelu(batch_norm(conv2d(h2, feature*8, name="h3"), "h3"))
            h4 = linear(tf.reshape(h3, [1,-1]), 1, "linear")
        return h4

    def discriminator(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv2D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0], kernel_size=self.filter_length)(input_tensor)
        cnn_layer_2 = Conv2D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[1], kernel_size=self.filter_length)(cnn_layer_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        layer_3 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer, activation=self.activation)(
            flattener)
        layer_4 = Dense(units=self.num_hidden[1], kernel_initializer=self.initializer, activation=self.activation)(
            layer_3)
        layer_out = Dense(units=self.labels_shape[0], kernel_initializer=self.initializer, activation=self.activation)(
            layer_4)
        dis_model = Model(inputs=input_tensor, outputs=layer_out)

        return dis_model

    def generator(self):
        pass

    def model(self):
        pass