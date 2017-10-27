# ---------------------------------------------------------#
#   astroNN.NN.cnn_models: Contain pre-define neural network architecture
# ---------------------------------------------------------#

import tensorflow as tf
from keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D, Dense, InputLayer, Flatten


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
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[1]))

    return model

def apogee_dnn_1(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: apogee_dnn_1
    PURPOSE: To create Dense Neural Network model 1 for apogee
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-25 Henry Leung
    """

    model = Sequential()
    model.add(InputLayer(batch_input_shape=input_shape))
    model.add(Dense(units=num_hidden[0], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_hidden[1], kernel_initializer=initializer, activation=activation))
    model.add(Dense(units=num_labels, activation="linear", input_dim=num_hidden[1]))

    return model
