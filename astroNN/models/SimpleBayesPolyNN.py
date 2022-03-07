# ---------------------------------------------------------#
#   astroNN.models.SimpleBayesPolyNN: Contain CNN Model
# ---------------------------------------------------------#
# from tensorflow import keras
# from astroNN.models.base_cnn import CNNBase
# from astroNN.nn.layers import BayesPolyFit
#
# Flatten, Input = keras.layers.Flatten, keras.layers.Input
# Model = keras.models.Model
# regularizers = keras.regularizers
#
#
# class SimpleBayesPolyNN(CNNBase):
#     """
#     Class for Neural Network for Gaia Polynomial fitting
#
#     :History: 2018-Jul-23 - Written - Henry Leung (University of Toronto)
#     """
#
#     def __init__(self, lr=0.005, init_w=None):
#         super().__init__()
#
#         self._implementation_version = '1.0'
#         self.max_epochs = 40
#         self.lr = lr
#         self.reduce_lr_epsilon = 0.00005
#         self.num_hidden = 3  # equals degree of polynomial to fit
#
#         self.reduce_lr_min = 1e-8
#         self.reduce_lr_patience = 2
#
#         self.input_norm_mode = 0
#         self.labels_norm_mode = 0
#         self.init_w = init_w
#
#         self.task = 'regression'
#         self.targetname = ['unbiased_parallax']
#
#     def model(self):
#         input_tensor = Input(shape=self._input_shape, name='input')
#         flattener = Flatten()(input_tensor)
#         output = BayesPolyFit(deg=self.num_hidden,
#                               output_units=self._labels_shape,
#                               use_xbias=True,
#                               name='output')(flattener)
#         model = Model(inputs=input_tensor, outputs=output)
#
#         return model
