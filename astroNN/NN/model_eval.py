# ---------------------------------------------------------#
#   astroNN.NN.model_eval: Evaluate CNN model
# ---------------------------------------------------------#
# import numpy as np
# from keras.models import load_model
# import astroNN.NN.train
#
#
# def hyperpara_search(h5name=None, target=None, test=True, model=None, num_hidden=None, num_filters=None, check_cannon=False,
#                  activation=None, initializer=None, filter_length=None, pool_length=None, batch_size=None,
#                  max_epochs=None, lr=None, early_stopping_min_delta=None, early_stopping_patience=None,
#                  reuce_lr_epsilon=None, reduce_lr_patience=None, reduce_lr_min=None, cnn_visualization=True,
#                  cnn_vis_num=None, test_noisy=None):
#     if h5name is None:
#         raise ValueError('Please specift the dataset name using h5name="...... "')
#     if target is None:
#         raise ValueError('Please specift a list of target names using target=[.., ...], target must be a list')
#     if model is None:
#         model = 'cnn_apogee_1'
#         print('No predefined model specified, using cnn_apogee_1 as default')
#     if num_hidden is None:
#         raise ValueError('Please specift a list of number of neurons using num_hidden=[.., ...], must be a list')
#     if activation is None:
#         activation = 'relu'
#         print('activation not provided, using default activation={}'.format(activation))
#     if initializer is None:
#         initializer = 'he_normal'
#         print('initializer not provided, using default initializer={}'.format(initializer))
#     if filter_length is None:
#         filter_length = 8
#         print('filter_length not provided, using default filter_length={}'.format(filter_length))
#     if pool_length is None:
#         pool_length = 4
#         print('pool_length not provided, using default pool_length={}'.format(pool_length))
#     if batch_size is None:
#         batch_size = 64
#         print('pool_length not provided, using default batch_size={}'.format(batch_size))
#     if max_epochs is None:
#         max_epochs = 200
#         print('max_epochs not provided, using default max_epochs={}'.format(max_epochs))
#     if lr is None:
#         lr = 1e-5
#         print('lr [Learning rate] not provided, using default lr={}'.format(lr))
#     if early_stopping_min_delta is None:
#         early_stopping_min_delta = 5e-6
#         print('early_stopping_min_delta not provided, using default early_stopping_min_delta={}'.format(lr))
#     if early_stopping_patience is None:
#         early_stopping_patience = 8
#         print('early_stopping_patience not provided, using default early_stopping_patience={}'.format(lr))
#     if reuce_lr_epsilon is None:
#         reuce_lr_epsilon = 7e-3
#         print('reuce_lr_epsilon not provided, using default reuce_lr_epsilon={}'.format(lr))
#     if reduce_lr_patience is None:
#         reduce_lr_patience = 2
#         print('reduce_lr_patience not provided, using default reduce_lr_patience={}'.format(lr))
#     if reduce_lr_min is None:
#         reduce_lr_min = 7e-08
#         print('reduce_lr_min not provided, using default reduce_lr_min={}'.format(lr))
#
#     model = astroNN.NN.train.apogee_train{h5name=h5name}:
