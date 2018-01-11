from abc import ABC, abstractmethod
import numpy as np

from keras.backend import clear_session
from keras.optimizers import Adam

from astroNN.models.loss.regression import mean_squared_error
from astroNN.models.loss.classification import categorical_cross_entropy, bayes_crossentropy_wrapper
from astroNN.models.NeuralNetMaster import NeuralNetMaster


class BayesianCNNBase(NeuralNetMaster, ABC):
    """Top-level class for a Bayesian convolutional neural network"""
    def __init__(self):
        """
        NAME:
            __init__on: {} \n".format(self._implementation_version))
            h.write("python versi
        PURPOSE:
            To define astroNN Bayesian convolutional neural network
        HISTORY:
            2018-Jan-06 - Written - Henry Leung (University of Toronto)
        """
        super(BayesianCNNBase, self).__init__()
        self.name = 'Bayesian Convolutional Neural Network'
        self._model_type = 'BCNN'
        self.initializer = None
        self.activation = None
        self._last_layer_activation = None
        self.num_filters = None
        self.filter_length = None
        self.pool_length = None
        self.num_hidden = None
        self.reduce_lr_epsilon = None
        self.reduce_lr_min = None
        self.reduce_lr_patience = None
        self.l2 = None
        self.inv_model_precision = None  # inverse model precision
        self.dropout_rate = 0.2
        self.length_scale = 0.01  # prior length scale
        self.mc_num = 10

        self.input_normalizer = None
        self.labels_normalizer = None
        self.input_norm_mode = 1
        self.labels_norm_mode = 2
        self.input_mean_norm = None
        self.input_std_norm = None
        self.labels_mean_norm = None
        self.labels_std_norm = None

    @abstractmethod
    def model(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, input_data, labels, inputs_err, labels_err):
        raise NotImplementedError

    @abstractmethod
    def test(self, input_data, inputs_err):
        raise NotImplementedError

    def compile(self):
        self.keras_model, variance_loss = self.model()

        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.optimizer_epsilon,
                                  decay=0.0)

        if self.task == 'regression':
            self._last_layer_activation = 'linear'
            self.keras_model.compile(loss={'output': mean_squared_error,
                                           'variance_output': variance_loss},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': 1., 'variance_output': .1})
        elif self.task == 'classification':
            print('Currently Not Working Properly')
            self._last_layer_activation = 'softmax'
            self.keras_model.compile(loss={'output': categorical_cross_entropy,
                                           'variance_output': bayes_crossentropy_wrapper(100, 10)},
                                     optimizer=self.optimizer,
                                     loss_weights={'output': 1., 'variance_output': .1})
        else:
            raise RuntimeError('Only "regression" and "classification" are supported')

        return None

    def pre_training_checklist_child(self):
        self.pre_training_checklist_master()

    def post_training_checklist_child(self):
        astronn_model = 'model_weights.h5'
        self.keras_model.save_weights(self.fullfilepath + astronn_model)
        print(astronn_model + ' saved to {}'.format(self.fullfilepath + astronn_model))

        np.savez(self.fullfilepath + '/astroNN_model_parameter.npz', id=self._model_identifier,
                 filterlen=self.filter_length, filternum=self.num_filters, hidden=self.num_hidden,
                 input=self.input_shape, labels=self.labels_shape, task=self.task, inv_tau=self.inv_model_precision,
                 input_mean=self.input_mean_norm,  labels_mean=self.labels_mean_norm, input_std=self.input_std_norm,
                 labels_std=self.labels_std_norm, targetname=self.targetname)

        clear_session()
