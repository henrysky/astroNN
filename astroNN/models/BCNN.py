# ---------------------------------------------------------#
#   astroNN.models.CNN: Contain CNN Model
# ---------------------------------------------------------#
import os
import numpy as np
import time

from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import MaxPooling1D, Conv1D, Dense, Dropout, Flatten, Activation
from keras.models import Model, Input
import keras.backend as K

from astroNN.models.utilities.normalizer import Normalizer
from astroNN.models.NeuralNetBases import BayesianCNNBase
from astroNN.models.utilities.generator import Bayesian_DataGenerator
from astroNN.models.loss.regression import mse_var_wrapper

class BCNN(BayesianCNNBase):
    """
    NAME:
        BCNN
    PURPOSE:
        To create Convolutional Neural Network model
    HISTORY:
        2017-Dec-21 - Written - Henry Leung (University of Toronto)
    """

    def __init__(self, lr=0.005):
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

        self._model_type = 'BCNN-MC'
        self._model_identifier = 'APOGEE_BCNN'
        self._implementation_version = '1.0'
        self.batch_size = 64
        self.initializer = 'he_normal'
        self.activation = 'relu'
        self.num_filters = [2, 4]
        self.filter_len = 8
        self.pool_length = 4
        self.num_hidden = [196, 96]
        self.max_epochs = 100
        self.lr = lr
        self.reduce_lr_epsilon = 0.00005

        self.reduce_lr_min = 1e-8
        self.reduce_lr_patience = 10
        self.data_normalization = True
        self.target = 'all'
        self.l2 = 1e-8

        self.task = 'regression'

    def model(self):
        input_tensor = Input(shape=self.input_shape)
        cnn_layer_1 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(input_tensor)
        activation_1 = Activation(activation=self.activation)(cnn_layer_1)
        dropout_1 = Dropout(self.dropout_rate)(activation_1)
        cnn_layer_2 = Conv1D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                             kernel_size=self.filter_length, kernel_regularizer=regularizers.l2(self.l2))(dropout_1)
        activation_2 = Activation(activation=self.activation)(cnn_layer_2)
        maxpool_1 = MaxPooling1D(pool_size=self.pool_length)(activation_2)
        flattener = Flatten()(maxpool_1)
        dropout_2 = Dropout(self.dropout_rate)(flattener)
        layer_3 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_2)
        activation_3 = Activation(activation=self.activation)(layer_3)
        dropout_3 = Dropout(self.dropout_rate)(activation_3)
        layer_4 = Dense(units=self.num_hidden[1], kernel_regularizer=regularizers.l2(self.l2),
                        kernel_initializer=self.initializer,
                        activation=self.activation)(dropout_3)
        activation_4 = Activation(activation=self.activation)(layer_4)
        linear_output = Dense(units=self.labels_shape, activation="linear", name='linear_output')(activation_4)
        variance_output = Dense(units=self.labels_shape, activation='linear', name='variance_output')(activation_4)

        model = Model(inputs=input_tensor, outputs=[linear_output, variance_output])

        mse_var_2 = mse_var_wrapper(linear_output)

        return model, mse_var_2

    def train(self, input_data, labels, inputs_err, labels_err):
        # Call the checklist to create astroNN folder and save parameters
        self.pre_training_checklist_child()

        self.input_normalizer = Normalizer(mode=self.input_norm_mode)
        self.labels_normalizer = Normalizer(mode=self.labels_norm_mode)

        norm_data, self.input_mean_norm, self.input_std_norm = self.input_normalizer.normalize(input_data)
        norm_labels, self.labels_mean_norm, self.labels_std_norm = self.labels_normalizer.normalize(labels)

        self.input_shape = (norm_data.shape[1], 1,)
        self.labels_shape = norm_labels.shape[1]

        self.compile()
        self.plot_model()

        csv_logger = CSVLogger(self.fullfilepath + 'log.csv', append=True, separator=',')

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=self.reduce_lr_epsilon,
                                      patience=self.reduce_lr_patience, min_lr=self.reduce_lr_min, mode='min',
                                      verbose=2)

        training_generator = Bayesian_DataGenerator(self.batch_size).generate(norm_data, norm_labels)

        self.keras_model.fit_generator(generator=training_generator,
                                       steps_per_epoch=norm_data.shape[0] // self.batch_size,
                                       epochs=self.max_epochs, max_queue_size=20, verbose=2, workers=os.cpu_count(),
                                       callbacks=[reduce_lr, csv_logger])

        # Call the post training checklist to save parameters
        self.post_training_checklist_child()

        return None

    def test(self, input_data, inputs_err):
        # Prevent shallow copy issue
        input_array = np.array(input_data)
        input_array -= self.input_mean_norm
        input_array /= self.input_std_norm
        input_array = np.atleast_3d(input_array)

        K.set_learning_phase(1)

        predictions = np.zeros((self.mc_num, input_array.shape[0], self.labels_shape))
        predictions_var = np.zeros((self.mc_num, input_array.shape[0], self.labels_shape))

        start_time = time.time()

        for counter, i in enumerate(range(self.mc_num)):
            if counter % 5 == 0:
                print('Completed {} of {} Monte Carlo, {:.03f} seconds elapsed'.format(counter, self.mc_num,
                                                                                       time.time() - start_time))
            input_array_with_error = input_array + np.random.normal(0, inputs_err)
            result = np.asarray(self.keras_model.predict(input_array_with_error))
            predictions[i] = result[0].reshape((input_array.shape[0], self.labels_shape))
            predictions_var[i] = result[1].reshape((input_array.shape[0], self.labels_shape))

        predictions *= self.labels_std_norm
        predictions += self.labels_mean_norm

        pred = np.mean(predictions, axis=0)
        var_mc_dropout = np.var(predictions, axis=0)

        var = np.mean(np.exp(predictions_var) * self.labels_std_norm, axis=0)
        pred_var = var + var_mc_dropout + self.inv_model_precision  # epistemic plus aleatoric uncertainty plus tau
        pred_var = np.sqrt(pred_var)

        print('Finished testing!')

        # self.aspcap_residue_plot(pred, y, pred_var)
        return pred, pred_var