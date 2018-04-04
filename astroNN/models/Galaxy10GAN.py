# ---------------------------------------------------------#
#   astroNN.models.Galaxy10GAN: Contain Galaxy10GAN model
# ---------------------------------------------------------#
from astroNN.config import keras_import_manager
from astroNN.models.CGANBase import CGANBase

keras = keras_import_manager()
regularizers = keras.regularizers
MaxPooling2D, Conv2D, Dense, Flatten, LeakyReLU, Input = keras.layers.MaxPooling2D, keras.layers.Conv2D, \
                                                         keras.layers.Dense, keras.layers.Flatten, \
                                                         keras.layers.LeakyReLU, keras.layers.Input
BatchNormalization, Dropout, Concatenate, Activation = keras.layers.BatchNormalization, keras.layers.Dropout, \
                                                       keras.layers.Concatenate, keras.layers.Activation
AvgPool2D, Reshape, UpSampling2D = keras.layers.AvgPool2D, keras.layers.Reshape, keras.layers.UpSampling2D
L1L2 = keras.regularizers.L1L2
Conv2DTranspose = keras.layers.Conv2DTranspose
TruncatedNormal, RandomNormal = keras.initializers.TruncatedNormal, keras.initializers.RandomNormal
Model = keras.models.Model


class Galaxy10GAN(CGANBase):
    """
    NAME:
        Galaxy10GAN
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
        super().__init__()

        self.name = 'Galaxy10GAN'
        self._implementation_version = '1.0'

        self.conv_initializer = TruncatedNormal(stddev=0.02)
        self.tran_conv_initializer = RandomNormal(stddev=0.02)
        self.num_filters = [64]
        self.filter_len = (4, 4)
        self.strides_length = (2, 2)
        self.weight_reg = lambda: L1L2(l1=1e-7, l2=1e-7)

    def discriminator(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv2D(64, 5, 5, border_mode='same', W_regularizer=self.weight_reg())(input_tensor)
        max_pool_1 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn_layer_1)
        lrele_1 = LeakyReLU(0.2)(max_pool_1)
        cnn_layer_2 = Conv2D(128, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(lrele_1)
        max_pool_2 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn_layer_2)
        lrele_2 = LeakyReLU(0.2)(max_pool_2)
        cnn_layer_3 = Conv2D(256, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(lrele_2)
        max_pool_3 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(cnn_layer_3)
        lrele_3 = LeakyReLU(0.2)(max_pool_3)
        cnn_layer_4 = Conv2D(1, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(lrele_3)
        avg_pool_4 = AvgPool2D(pool_size=(4, 4), border_mode='valid')(cnn_layer_4)
        flatten = Flatten()(avg_pool_4)
        dis_poutput = Activation('sigmoid')(flatten)
        model = Model(input=input_tensor, outputs=dis_poutput)
        return model

    def generator(self):
        input_tensor = Input(shape=self.input_shape)
        dense_1 = Dense(256 * 4 * 4, input_shape=(100,), W_regularizer=self.weight_reg())(input_tensor)
        BN_1 = BatchNormalization(mode=0)(dense_1)
        reshape_1 = Reshape([256, 4, 4])(BN_1)
        cnn_layer_1 = Conv2D(128, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(reshape_1)
        BN_2 = BatchNormalization(mode=0, axis=1)(cnn_layer_1)
        lrele_1 = LeakyReLU(0.2)(BN_2)
        upsample_1 = UpSampling2D(size=(2, 2))(lrele_1)
        cnn_layer_2 = Conv2D(128, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(upsample_1)
        BN_3 = BatchNormalization(mode=0, axis=1)(cnn_layer_2)
        lrele_2 = LeakyReLU(0.2)(BN_3)
        upsample_2 = UpSampling2D(size=(2, 2))(lrele_2)
        cnn_layer_3 = Conv2D(64, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(upsample_2)
        BN_4 = BatchNormalization(mode=0, axis=1)(cnn_layer_3)
        lrele_3 = LeakyReLU(0.2)(BN_4)
        upsample_3 = UpSampling2D(size=(2, 2))(lrele_3)
        cnn_layer_4 = Conv2D(3, 3, 3, border_mode='same', W_regularizer=self.weight_reg())(upsample_3)
        gen_output = Activation('sigmoid')(cnn_layer_4)
        model = Model(inputs=input_tensor, outputs=gen_output)
        return model
