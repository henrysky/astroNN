# ---------------------------------------------------------#
#   astroNN.NN.cnn_models: Contain pre-define neural network architecture
# ---------------------------------------------------------#

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling1D, Conv1D, Dense, InputLayer, Flatten


def cnn_model_1(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
    """
    NAME: cnn_model_1
    PURPOSE: To create Convolutional Neural Network model 1
    INPUT:
    OUTPUT: the model
    HISTORY:
        2017-Oct-14 Henry Leung
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

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