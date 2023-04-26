.. automodule:: astroNN.models.apogee_models

Encoder-decoder for APOGEE and Kepler - **ApokascEncoderDecoder**
----------------------------------------------------------------------------------------------------------

.. autoclass:: astroNN.models.apogee_models.ApokascEncoderDecoder
    :members:

.. inheritance-diagram:: astroNN.models.apogee_models.ApokascEncoderDecoder
   :parts: 1

`ApokascEncoderDecoder` can only be used with Apogee spectra with 7,514 pixels and Kepler PSD with 2,092. Both numbers are **hardcoded** into the model

Please refers to the paper https://ui.adsabs.harvard.edu/abs/2023arXiv230205479L/abstract and https://github.com/henrysky/astroNN_ages for detail

.. code-block:: python

    from astroNN.models import ApokascEncoderDecoder
    from astroNN.datasets import H5Loader

    # Load the train data from dataset first, x_train is spectra and y_train will be ASPCAP labels
    loader = H5Loader('datasets.h5')
    loader.load_combined = True
    loader.load_err = True
    x_train, y_train, x_err, y_err = loader.load()

    # And then create an instance of Bayesian Convolutional Neural Network class
    ved = ApokascEncoderDecoder()

    # You don't have to specify the task because its 'regression' by default. But if you are doing classification. you can set task='classification'
    ved.task = 'regression'

    # Set max_epochs to 10 for a quick result. You should train more epochs normally, especially with dropout
    ved.max_epochs = 10
    ved.train(x_train, y_train, x_err, y_err)


Here is a list of parameter you can set but you can also not set them to use default

.. code-block:: python

    ved.batch_size = 128
    ved.initializer = 'glorot_uniform'
    ved.activation = 'relu'
    ved.num_filters = [32, 64, 16, 16]
    ved.filter_len = [8, 32]
    ved.pool_length = 2
    ved.num_hidden = [16, 16]
    ved.latent_dim = 5
    ved.max_epochs = 100
    ved.lr = 0.005
    ved.reduce_lr_epsilon = 0.00005
    ved.reduce_lr_min = 0.0000000001
    ved.reduce_lr_patience = 10
    ved.target = 'PSD'
    ved.l2 = 5e-9
    ved.input_norm_mode = 2
    ved.labels_norm_mode = 0

.. note:: You can disable astroNN data normalization via ``ApokascEncoderDecoder.input_norm_mode=0`` as well as ``ApokascEncoderDecoder.labels_norm_mode=0`` and do normalization yourself. But make sure you don't normalize labels with MAGIC_NUMBER (missing labels).

After the training, you can use `ved` in this case and call test method to test the neural network on test data. Or you can load the folder by

.. code-block:: python

    from astroNN.models import load_folder
    ved = load_folder('astroNN_0101_run001')

    # Load the test data from dataset, x_test is APOGEE spectra
    # something here

    # pred contains denormalized result aka. Kepler PSD prediction in this case
    pred = ved.test(x_test)
    
    # methods like predict_encoder() and predict_decoder() also available
