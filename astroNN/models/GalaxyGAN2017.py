# ---------------------------------------------------------#
#   astroNN.models.GalaxyGAM2017: Contain GalaxyGAN2017 model
# ---------------------------------------------------------#
from keras.layers import Conv2DTranspose, Conv2D, Dense, Flatten, LeakyReLU, BatchNormalization
from keras.models import Model, Input
from keras.initializers import TruncatedNormal, RandomNormal

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

        self.conv_initializer = TruncatedNormal(stddev=0.02)
        self.tran_conv_initializer = RandomNormal(stddev=0.02)
        self.num_filters = [64]
        self.filter_length = (4,4)
        self.strides_length = (2,2)

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

    def discriminator(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_length, strides=self.strides_length)(input_tensor)
        leaky_1 = LeakyReLU(alpha=0.2)(cnn_layer_1)
        cnn_layer_2 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*2,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_1)
        BN_1 = BatchNormalization()(cnn_layer_2)
        leaky_2 = LeakyReLU(alpha=0.2)(BN_1)
        cnn_layer_3 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*4,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_2)
        BN_2 = BatchNormalization()(cnn_layer_3)
        leaky_3 = LeakyReLU(alpha=0.2)(BN_2)
        cnn_layer_4 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_3)
        BN_3 = BatchNormalization()(cnn_layer_4)
        leaky_4 = LeakyReLU(alpha=0.2)(BN_3)
        flattener_1 = Flatten()(leaky_4)
        discriminator_output = Dense(activation='linear')(flattener_1)

        return discriminator_output

    def jaskda(self, cond):

            d1 = deconv2d(tf.nn.relu(e8), [1,num[1],num[1],feature*8], name="d1")
            d1 = tf.concat([tf.nn.dropout(batch_norm(d1, "d1"), 0.5), e7], 3)
            d2 = deconv2d(tf.nn.relu(d1), [1,num[2],num[2],feature*8], name="d2")
            d2 = tf.concat([tf.nn.dropout(batch_norm(d2, "d2"), 0.5), e6], 3)
            d3 = deconv2d(tf.nn.relu(d2), [1,num[3],num[3],feature*8], name="d3")
            d3 = tf.concat([tf.nn.dropout(batch_norm(d3, "d3"), 0.5), e5], 3)
            d4 = deconv2d(tf.nn.relu(d3), [1,num[4],num[4],feature*8], name="d4")
            d4 = tf.concat([batch_norm(d4, "d4"), e4], 3)
            d5 = deconv2d(tf.nn.relu(d4), [1,num[5],num[5],feature*4], name="d5")
            d5 = tf.concat([batch_norm(d5, "d5"), e3], 3)
            d6 = deconv2d(tf.nn.relu(d5), [1,num[6],num[6],feature*2], name="d6")
            d6 = tf.concat([batch_norm(d6, "d6"), e2], 3)
            d7 = deconv2d(tf.nn.relu(d6), [1,num[7],num[7],feature], name="d7")
            d7 = tf.concat([batch_norm(d7, "d7"), e1], 3)
            d8 = deconv2d(tf.nn.relu(d7), [1,num[8],num[8],conf.img_channel], name="d8")

            return tf.nn.tanh(d8)

    def generator(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_length, strides=self.strides_length)(input_tensor)
        leaky_1 = LeakyReLU(alpha=0.2)(cnn_layer_1)
        cnn_layer_2 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*2,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_1)
        BN_1 = BatchNormalization()(cnn_layer_2)
        leaky_2 = LeakyReLU(alpha=0.2)(BN_1)
        cnn_layer_3 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*4,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_2)
        BN_2 = BatchNormalization()(cnn_layer_3)
        leaky_3 = LeakyReLU(alpha=0.2)(BN_2)
        cnn_layer_4 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_3)
        BN_3 = BatchNormalization()(cnn_layer_4)
        leaky_4 = LeakyReLU(alpha=0.2)(BN_3)
        cnn_layer_5 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_4)
        BN_4= BatchNormalization()(cnn_layer_5)
        leaky_5 = LeakyReLU(alpha=0.2)(BN_3)
        cnn_layer_6 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_5)
        BN_5 = BatchNormalization()(cnn_layer_6)
        leaky_6 = LeakyReLU(alpha=0.2)(BN_5)
        cnn_layer_7 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_6)
        BN_6 = BatchNormalization()(cnn_layer_7)
        leaky_7 = LeakyReLU(alpha=0.2)(BN_6)
        cnn_layer_8 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_7)
        BN_7 = BatchNormalization()(cnn_layer_8)
        leaky_8 = LeakyReLU(alpha=0.2)(BN_7)

        size = conf.img_size
        num = [0] * 9
        for i in range(1, 9):
            num[9 - i] = size
            size = (size + 1) / 2

        tcnn_layer_1 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=self.num_filters[0]*8, kernel_size=self.filter_length,
                                       strides=self.strides_length)

    def model(self):
        pass