# ---------------------------------------------------------#
#   astroNN.NN.cnn_models: Contain pre-define neural network architecture
# ---------------------------------------------------------#

from keras.layers import MaxPooling1D, Conv1D, Dense, InputLayer, Flatten, GaussianNoise, concatenate, Dropout
from keras.models import Sequential, Model, Input
from keras import regularizers


def apogee_cnn_1(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: apogee_cnn_1
    PURPOSE: To create Convolutional Neural Network model 1 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                     kernel_size=filter_length))
    model.add(Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[1],
                     kernel_size=filter_length))
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Flatten())
    model.add(GaussianNoise(0.01))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[-1]))
    return model


def apogee_cnn_2(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: apogee_cnn_2
    PURPOSE: To create Convolutional Neural Network model 2 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-27 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                     kernel_size=filter_length))
    model.add(Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[1],
                     kernel_size=filter_length))
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Flatten())
    model.add(GaussianNoise(0.01))
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[2], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[-1]))
    return model


def apogee_cnn_3(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: apogee_cnn_3
    PURPOSE: To create Convolutional Neural Network model 3 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-31 Henry Leung
    """
    input_shape = Input(shape=(0, 7514))

    tower_1 = Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                     kernel_size=filter_length)(input_shape)
    tower_1 = MaxPooling1D(pool_size=pool_length, padding='same')(tower_1)

    tower_2 = Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                     kernel_size=filter_length)(input_shape)
    tower_2 = MaxPooling1D(pool_size=pool_length, padding='same')(tower_2)

    tower_3 = Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                     kernel_size=filter_length)(input_shape)
    tower_3 = MaxPooling1D(pool_size=pool_length, padding='same')(tower_3)

    merged = concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = Flatten()(merged)
    out = Dense(num_hidden[0], activation='relu')(merged)
    out = Dense(num_labels, activation='softmax')(out)

    model = Model(input_shape, out)
    return model

    # tower_3 = Sequential()
    # tower_3.add(InputLayer(batch_input_shape=input_shape))
    # tower_3.add(Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
    #            kernel_size=filter_length))
    # tower_3.add(MaxPooling1D(pool_size=pool_length, padding='same'))

    # model =  Concatenate([tower_1, tower_2, tower_3])
    # model.add(Flatten())
    # model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    # model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    # model.add(Dense(units=num_hidden[2], kernel_initializer=initializer, activation=activation))
    # model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[-1]))
    # return model


def apogee_generative_1(input_shape, initializer, activation, num_hidden):
    """
    NAME: apogee_generative_1
    PURPOSE: To create Generative Neural Network model 1 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-28 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    # Layer 2 should have no more than 32 neurones
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[2], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=input_shape[1], activation="linear", input_dim=num_hidden[-1]))
    return model


def apogee_generator_1(input_shape, initializer, activation, num_hidden):
    """
    NAME: apogee_generative_1
    PURPOSE: To create Generative Neural Network model 1 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-28 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    # Layer 2 should have no more than 32 neurones
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[2], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=7514, activation="linear", input_dim=num_hidden[-1]))
    return model


def gaia_dnn_1(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: apogee_cnn_2
    PURPOSE: To create Convolutional Neural Network model 2 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-27 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[2], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[-1]))
    return model


def gaia_cnn_1(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: apogee_cnn_1
    PURPOSE: To create Convolutional Neural Network model 1 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-14 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Conv1D(kernel_initializer=initializer, activation=activation, padding="same", filters=num_filters[0],
                     kernel_size=filter_length))
    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Flatten())
    # model.add(Dropout(0.1))
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    # model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[-1]))
    return model
