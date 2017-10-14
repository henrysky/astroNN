# ---------------------------------------------------------#
#   astroNN.NN.cnn_models: Contain pre-define neural network architecture
# ---------------------------------------------------------#

import tensorflow as tf
from tensorflow.python.keras.models  import Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import MaxPooling1D, Conv1D, Dense, InputLayer, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam


def cnn_model1(input_shape, initializer, activation, num_filters, filter_length, pool_length, num_hidden, num_labels):
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

    # Default loss function parameters
    early_stopping_min_delta = 0.0001
    early_stopping_patience = 4
    reduce_lr_factor = 0.5
    reuce_lr_epsilon = 0.0009
    reduce_lr_patience = 2
    reduce_lr_min = 0.00008

    # loss function to minimize
    loss_function = 'mean_squared_error'

    # compute accuracy and mean absolute deviation
    metrics = ['accuracy', 'mae']

    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

    tbCallBack = TensorBoard(log_dir=datadir + 'logs', histogram_freq=0, batch_size=32, write_graph=True,
                             write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                   patience=early_stopping_patience, verbose=2, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reuce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
