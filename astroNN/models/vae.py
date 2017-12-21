# ---------------------------------------------------------#
#   astroNN.models.var: Contain Variational Autoencoder Model
# ---------------------------------------------------------#


from keras.layers import MaxPooling1D, Conv1D, Dense, InputLayer, Flatten, GaussianNoise, concatenate, Dropout, Activation
from keras.models import Sequential, Model, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers


class VAE(object):
    def __init__(self):
        print("VAE")

    def model(self, input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
        """
        NAME:
            apogee_cnn_1
        PURPOSE:
            To create Convolutional Neural Network model 1 for apogee
        INPUT:
        OUTPUT: the model
        HISTORY:
            2017-Oct-14 Henry Leung
        """
        input_tensor = Input(batch_shape=input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                         kernel_size=filter_length,kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
        dropout_1 = Dropout(0.3)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                         kernel_size=filter_length,kernel_regularizer=regularizers.l2(1e-4))(dropout_1)
        maxpool_1 = MaxPooling1D(pool_size=pool_length)(cnn_layer_2)
        dropout_2 = Dropout(0.3)(maxpool_1)
        layer_3 = Dense(units=num_hidden[1], kernel_regularizer=regularizers.l2(1e-4), kernel_initializer='he_normal',
                        activation='relu')(dropout_2)
        dropout_3 = Dropout(0.3)(layer_3)
        layer_4 = Dense(units=num_hidden[1], kernel_regularizer=regularizers.l2(1e-4), kernel_initializer='he_normal',
                        activation='relu')(dropout_3)
        linear_output = Dense(units=1, activation="linear", name='linear_output')(layer_4)
        variance_output = Dense(units=1, activation='linear', name='variance_output')(layer_4)

        model = Model(inputs=input_tensor, outputs=[variance_output, linear_output])

        return model, linear_output, variance_output