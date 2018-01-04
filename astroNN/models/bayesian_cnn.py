# ---------------------------------------------------------#
#   astroNN.models.bayesian: Contain CNN Model
# ---------------------------------------------------------#
import os
import time

import keras.backend as K
import numpy as np
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import MaxPooling1D, Conv1D, Dense, Dropout, Flatten, Layer, RepeatVector
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed

from astroNN.models.models_shared import ModelStandard
from astroNN.models.models_tools import threadsafe_generator

K.set_learning_phase(1)


class BCNN(ModelStandard):
    """
    NAME:
        BCNN
    PURPOSE:
        To create Bayesian Convolutional Neural Network model, this the implementation of StarNet with arXiv:1506.02158
    HISTORY:
        2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self):
        """
        NAME:
            model
        PURPOSE:
            To create Convolutional Neural Network model
        INPUT:
        OUTPUT:
        HISTORY:
            2017-Dec-21 - Written - Henry Leung (University of Toronto)
        """
        super(BCNN, self).__init__()

        self.name = 'Bayesian Convolutional Neural Network with Variational Inference'
        self._model_type = 'BCNN-MC'
        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_length = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.max_epochs = 250
        self.lr = 0.005
        self.reduce_lr_epsilon = 0.00005
        self.reduce_lr_min = 0.0000000001
        self.reduce_lr_patience = 10
        self.fallback_cpu = False
        self.limit_gpu_mem = True
        self.data_normalization = True
        self.target = 'all'
        self.l2 = 1e-7
        self.dropout_rate = 0.2
        self.length_scale = 0.05  # prior length scale
        self.inv_model_precision = 0.0  # inverse model precision
        self.mc_num = 10

    def model(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        dropout_1 = Dropout(self.dropout_rate)(cnn_layer_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, activation=self.activation, padding="same",
                             filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(cnn_layer_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = Dropout(self.dropout_rate)(flattener)
        layer_3 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_2)
        dropout_3 = Dropout(self.dropout_rate)(layer_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_3)
        linear_output = Dense(units=self.output_shape[0], activation="linear", name='linear_output')(layer_4)
        variance_output = Dense(units=self.output_shape[0], activation='linear', name='variance_output')(layer_4)

        model = Model(inputs=input_tensor, outputs=[linear_output, variance_output])

        mse_var_2 = self.mse_var_wrapper(linear_output)

        return model, mse_var_2

    def compile(self):
        self.keras_model, mse_var_2 = self.model()

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.task == 'regression':
            self.keras_model.compile(loss={'linear_output': self.mean_squared_error,
                                           'variance_output': mse_var_2},
                                     optimizer=self.optimizer,
                                     loss_weights={'linear_output': 1., 'variance_output': .1})
        elif self.task == 'classification':
            print('Currently Not Working Properly')
            self.keras_model.compile(loss={'linear_output': self.categorical_cross_entropy,
                                           'variance_output': self.bayes_crossentropy_wrapper(100, 10)},
                                     optimizer=self.optimizer,
                                     loss_weights={'linear_output': 1., 'variance_output': .1})
        return None

    def train(self, x_data, y_data):
        x_data_norm, y_data_norm = super().train(x_data, y_data)

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)
        self.compile()
        self.plot_model()

        self.inv_model_precision = (2*y_data_norm.shape[1]*self.l2) / (self.length_scale**2 * (1-self.dropout_rate))

        np.save(self.fullfilepath + 'astroNN_use_only/inv_tau.npy', self.inv_model_precision)

        training_generator = DataGenerator(x_data_norm.shape[1], self.batch_size).generate(x_data_norm, y_data_norm)

        self.keras_model.fit_generator(generator=training_generator, steps_per_epoch=x_data_norm.shape[0] // self.batch_size,
                                       epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                                       callbacks=[reduce_lr, csv_logger])

        astronn_model = 'model_weights.h5'
        self.keras_model.save_weights(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        K.clear_session()

        return None

    def test(self, x):
        x = super().test(x)

        predictions = np.zeros((self.mc_num, x.shape[0], self.output_shape[0]))
        predictions_var = np.zeros((self.mc_num, x.shape[0], self.output_shape[0]))

        start_time = time.time()

        for counter, i in enumerate(range(self.mc_num)):
            if counter % 5 == 0:
                print('Completed {} of {} Monte Carlo, {:.03f} seconds elapsed'.format(counter, self.mc_num,
                                                                                       time.time() - start_time))
            result = np.asarray(self.keras_model.predict(x))
            predictions[i] = result[0].reshape((x.shape[0], self.output_shape[0]))
            predictions_var[i] = result[1].reshape((x.shape[0], self.output_shape[0]))

        # get mean results and its varience and mean unceratinty from dropout
        mu_std = np.load(self.fullfilepath + '/meanstd.npy')

        pred = np.mean(predictions, axis=0)
        var_mc_dropout = np.var(predictions, axis=0)

        pred *= mu_std[1]
        pred += mu_std[0]
        var_mc_dropout *= mu_std[1]

        var = np.mean(np.exp(predictions_var)* mu_std[1], axis=0)
        pred_var = var + var_mc_dropout + self.inv_model_precision  # epistemic plus aleatoric uncertainty plus tau

        print('Finished testing!')

        # self.aspcap_residue_plot(pred, y, pred_var)
        return pred, pred_var

    def create_epistemic_uncertainty_model(self, epistemic_monte_carlo_simulations):
        inpt = Input(shape=(self.keras_model.input_shape[1:]))
        x = RepeatVector(epistemic_monte_carlo_simulations)(inpt)
        # Keras TimeDistributed can only handle a single output from a model :(
        # and we technically only need the softmax outputs.
        hacked_model = Model(inputs=self.keras_model.inputs, outputs=self.keras_model.outputs[0])
        x = TimeDistributed(hacked_model, name='epistemic_monte_carlo')(x)
        # predictive probabilities for each class
        mean = TimeDistributedMean(name='epistemic')(x)
        variance = PredictiveLinear(name='epistemic_variance')(mean)
        epistemic_model = Model(inputs=inpt, outputs=[variance, mean])

        return epistemic_model

    # # 1. Load the model
    # # 2. compile the model
    # # 3. Set learning phase to train
    # # 4. predict
    # def predict(self, x, y):
    #     model = self.create_epistemic_uncertainty_model(10)
    #     x = super().test(x)
    #
    #     # set learning phase to 1 so that Dropout is on. In keras master you can set this
    #     # on the TimeDistributed layer
    #     K.set_learning_phase(1)
    #
    #     master = model.predict(x)
    #
    #     self.aspcap_residue_plot(master[0], y, master[1])


class DataGenerator(object):
    """
    NAME:
        DataGenerator
    PURPOSE:
        To generate data for Keras
    INPUT:
    OUTPUT:
    HISTORY:
        2017-Dec-02 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, dim, batch_size, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle is True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, spectra, labels, list_IDs_temp):
        'Generates data of batch_size samples'
        # X : (n_samples, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, 1))
        y = np.empty((self.batch_size, labels.shape[1]))

        # Generate data
        X[:, :, 0] = spectra[list_IDs_temp]
        y[:] = labels[list_IDs_temp]

        return X, y

    @threadsafe_generator
    def generate(self, input, labels):
        'Generates batches of samples'
        # Infinite loop
        list_IDs = range(input.shape[0])
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = indexes[i * self.batch_size:(i + 1) * self.batch_size]

                # Generate data
                X, y = self.__data_generation(input, labels, list_IDs_temp)

                yield (X, {'linear_output': y, 'variance_output': y})


# Take a mean of the results of a TimeDistributed layer.
# Applying TimeDistributedMean()(TimeDistributed(T)(x)) to an
# input of shape (None, ...) returns output of same size.
class TimeDistributedMean(Layer):
    def build(self, input_shape):
        super(TimeDistributedMean, self).build(input_shape)

    # input shape (None, T, ...)
    # output shape (None, ...)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]

    def call(self, x):
        return K.mean(x, axis=1)


# Apply the predictive entropy function for input with C output.
# Input of shape (None, C, ...) returns output with shape (None, ...)
# Input should be predictive means for the C classes.
# In the case of a single classification, output will be (None,).
class PredictiveLinear(Layer):
    def build(self, input_shape):
        super(PredictiveLinear, self).build(input_shape)

    # input shape (None, C, ...)
    # output shape (None, ...)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

    # x - prediction probability for each class(C)
    def call(self, x):
        return -1 * K.sum(x, axis=1)

class PredictiveEntropy(Layer):
    def build(self, input_shape):
        super(PredictiveEntropy, self).build(input_shape)

    # input shape (None, C, ...)
    # output shape (None, ...)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

    # x - prediction probability for each class(C)
    def call(self, x):
        return -1 * K.sum(K.log(x) * x, axis=1)

