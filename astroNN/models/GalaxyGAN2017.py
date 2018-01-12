# ---------------------------------------------------------#
#   astroNN.models.GalaxyGAM2017: Contain GalaxyGAN2017 model
# ---------------------------------------------------------#
from keras.layers import Conv2DTranspose, Conv2D, Dense, Flatten, LeakyReLU, BatchNormalization, Dropout, Concatenate, \
    Activation
from keras.models import Model, Input
from keras.initializers import TruncatedNormal, RandomNormal

from astroNN.models.CGANBase import CGANBase


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

        self.name = 'GalaxyGAN (arXiv:1702.00403)'
        self._model_identifier = 'GalaxyGAN2017'
        self._implementation_version = '1.0'

        self.conv_initializer = TruncatedNormal(stddev=0.02)
        self.tran_conv_initializer = RandomNormal(stddev=0.02)
        self.num_filters = [64]
        self.filter_length = (4,4)
        self.strides_length = (2,2)
        self.img_size = 424

        print('Currently NOT WORKING!!!!!!!!')

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
        BN_1 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_2)
        leaky_2 = LeakyReLU(alpha=0.2)(BN_1)
        cnn_layer_3 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*4,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_2)
        BN_2 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_3)
        leaky_3 = LeakyReLU(alpha=0.2)(BN_2)
        cnn_layer_4 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_3)
        BN_3 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_4)
        leaky_4 = LeakyReLU(alpha=0.2)(BN_3)
        flattener_1 = Flatten()(leaky_4)
        discriminator_output = Dense(activation='linear')(flattener_1)

        model = Model(input=input_tensor, outputs=discriminator_output)

        return model

    def generator(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_length, strides=self.strides_length)(input_tensor)
        leaky_1 = LeakyReLU(alpha=0.2)(cnn_layer_1)
        cnn_layer_2 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*2,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_1)
        BN_1 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_2)
        leaky_2 = LeakyReLU(alpha=0.2)(BN_1)
        cnn_layer_3 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*4,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_2)
        BN_2 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_3)
        leaky_3 = LeakyReLU(alpha=0.2)(BN_2)
        cnn_layer_4 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_3)
        BN_3 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_4)
        leaky_4 = LeakyReLU(alpha=0.2)(BN_3)
        cnn_layer_5 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_4)
        BN_4= BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_5)
        leaky_5 = LeakyReLU(alpha=0.2)(BN_4)
        cnn_layer_6 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_5)
        BN_5 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_6)
        leaky_6 = LeakyReLU(alpha=0.2)(BN_5)
        cnn_layer_7 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_6)
        BN_6 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_7)
        leaky_7 = LeakyReLU(alpha=0.2)(BN_6)
        cnn_layer_8 = Conv2D(kernel_initializer=self.conv_initializer, padding="same", filters=self.num_filters[0]*8,
                             kernel_size=self.filter_length, strides=self.strides_length)(leaky_7)
        BN_7 = BatchNormalization(epsilon=1e-5, momentum=0.1)(cnn_layer_8)

        size = self.img_size
        num = [0] * 9
        for i in range(1, 9):
            num[9 - i] = size
            size = (size + 1) / 2

        leaky_8 = LeakyReLU(alpha=0.0)(BN_7)
        tcnn_layer_1 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[1], num[1], self.num_filters[0]*8],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_8)
        BN_8 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_1)
        dropout_1 = Dropout(0.5)(BN_8)
        con_layer_1 = Concatenate([dropout_1, BN_6])
        leaky_9 = LeakyReLU(alpha=0.0)(con_layer_1)
        tcnn_layer_2 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[2], num[2], self.num_filters[0]*8],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_9)
        BN_9 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_2)
        dropout_2 = Dropout(0.5)(BN_9)
        con_layer_2 = Concatenate([dropout_2, BN_5])
        leaky_10 = LeakyReLU(alpha=0.0)(con_layer_2)
        tcnn_layer_3 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[3], num[3], self.num_filters[0]*8],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_10)
        BN_10 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_3)
        dropout_3 = Dropout(0.5)(BN_10)
        con_layer_3 = Concatenate([dropout_3, BN_4])
        leaky_11 = LeakyReLU(alpha=0.0)(con_layer_3)
        tcnn_layer_4 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[4], num[4], self.num_filters[0]*8],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_11)
        BN_11 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_4)
        dropout_4 = Dropout(0.5)(BN_11)
        con_layer_4 = Concatenate([dropout_4, BN_3])
        leaky_12 = LeakyReLU(alpha=0.0)(con_layer_4)
        tcnn_layer_5 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[5], num[5], self.num_filters[0]*4],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_12)
        BN_12 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_5)
        dropout_5 = Dropout(0.5)(BN_12)
        con_layer_5 = Concatenate([dropout_5, BN_2])
        leaky_13 = LeakyReLU(alpha=0.0)(con_layer_5)
        tcnn_layer_6 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[6], num[6], self.num_filters[0]*2],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_13)
        BN_13 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_6)
        dropout_6 = Dropout(0.5)(BN_13)
        con_layer_6 = Concatenate([dropout_6, BN_1])
        leaky_14 = LeakyReLU(alpha=0.0)(con_layer_6)
        tcnn_layer_7 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[7], num[7], self.num_filters[0]],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_14)
        BN_14 = BatchNormalization(epsilon=1e-5, momentum=0.1)(tcnn_layer_7)
        dropout_7 = Dropout(0.5)(BN_14)
        con_layer_7 = Concatenate([dropout_7, cnn_layer_1])
        leaky_15 = LeakyReLU(alpha=0.0)(con_layer_7)
        tcnn_layer_8 = Conv2DTranspose(kernel_constraint=self.tran_conv_initializer, padding="same",
                                       filters=[1, num[8], num[8], self.num_filters[0]],
                                       kernel_size=self.filter_length, strides=self.strides_length)(leaky_15)
        final_out = Activation('tanh')(tcnn_layer_8)


        model = Model(inputs=input_tensor, outputs=final_out)

        return model

    def model(self):
        pass