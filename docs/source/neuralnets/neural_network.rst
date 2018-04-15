Introduction to astroNN Neural Nets
=======================================================

Available astroNN Neural Net Classes
--------------------------------------

All astroNN Neural Nets are inherited from some child classes which inherited NeuralNetMaster, NeuralNetMaster also
relies relies on two major component, `Normalizer` and `GeneratorMaster`

::

    Normalizer (astroNN.nn.utilities.normalizer.Normalizer)

    GeneratorMaster (astroNN.nn.utilities.generator.GeneratorMaster)
    ├── CNNDataGenerator
    ├── Bayesian_DataGenerator
    └── CVAE_DataGenerator

    NeuralNetMaster (astroNN.models.NeuralNetMaster.NeuralNetMaster)
    ├── CNNBase
    │   ├── ApogeeCNN
    │   ├── StarNet2017
    │   └── Cifar10CNN
    ├── BayesianCNNBase
    │   ├── MNIST_BCNN  # For authors testing only
    │   └── ApogeeBCNN
    ├── ConvVAEBase
    │   └── APGOEECVAE  # For authors testing only
    └── CGANBase
        └── GalaxyGAN2017  # For authors testing only

Creating Your Own Model with astroNN Neural Net Classes
----------------------------------------------------------

You can create your own neural network model inherits from astroNN Neural Network class to take advantage of the existing
code in this package. Here we will go throught how to create a simple model to do classification with MNIST dataset with
one convolutional layer and one fullly connected layer neural network.

Lets create a python script named `custom_models.py` under an arbitrary folder, lets say `~/` which is your home folder,
add ``~/custom_models.py`` to astroNN configuration file.

.. code-block:: python

    # import everything we need

    # astroNN keras_import_manager will import tf.keras iautomatically if keras is not detected
    from astroNN.config import keras_import_manager
    # this is the astroNN neural net abstract class we will going to inherit from
    from astroNN.models.CNNBase import CNNBase

    keras = keras_import_manager()
    regularizers = keras.regularizers
    MaxPooling2D, Conv2D, Dense, Flatten, Activation, Input = keras.layers.MaxPooling2D, keras.layers.Conv2D, \
                                                              keras.layers.Dense, keras.layers.Flatten, \
                                                              keras.layers.Activation, keras.layers.Input

    # now we are creating a custom model based on astroNN neural net abstract class
    class my_custom_model(CNNBase):
        def __init__(self, lr=0.005):
            # standard super for inheriting abstrack class
            super().__init__()

            # some default hyperparameters
            self._implementation_version = '1.0'
            self.initializer = 'he_normal'
            self.activation = 'relu'
            self.num_filters = [8]
            self.filter_len = (3, 3)
            self.pool_length = (4, 4)
            self.num_hidden = [128]
            self.max_epochs = 1
            self.lr = lr
            self.reduce_lr_epsilon = 0.00005

            self.task = 'classification'
            # you should set the targetname some that you know what those output neurones are representing
            # in this case the outpu the neurones are simply representing digits
            self.targetname = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

            # set default input norm mode to 255 to normalize images correctly
            self.input_norm_mode = 255
            # set default labels norm mode to 0 (equivalent to do nothing) to normalize labels correctly
            self.labels_norm_mode = 0

        def model(self):
            input_tensor = Input(shape=self.input_shape, name='input')
            cnn_layer_1 = Conv2D(kernel_initializer=self.initializer, padding="same", filters=self.num_filters[0],
                                 kernel_size=self.filter_len)(input_tensor)
            activation_1 = Activation(activation=self.activation)(cnn_layer_1)
            maxpool_1 = MaxPooling2D(pool_size=self.pool_length)(activation_1)
            flattener = Flatten()(maxpool_1)
            layer_2 = Dense(units=self.num_hidden[0], kernel_initializer=self.initializer)(flattener)
            activation_2 = Activation(activation=self.activation)(layer_2)
            layer_3 = Dense(units=self.labels_shape, kernel_initializer=self.initializer)(activation_2)
            output = Activation(activation=self._last_layer_activation, name='output')(layer_3)

            model = Model(inputs=input_tensor, outputs=output)

            return model

Save the file and we can open python under the same location as the python script

.. code-block:: python

    # import everything we need
    from custom_models import my_custom_model
    from keras.datasets import mnist
    from keras.utils import np_utils

    # load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert to approach type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)

    # create a neural network instance
    net = my_custom_model()

    # train
    net.train(x_train, y_train)

    # save the model after training
    net.save("trained_models_folder")

If you want to share the trained models, you have to copy `custom_models.py` to the inside of the folder so that
astroNN can load it successfully on other computers.

The second way is you send the file which is `custom_models.py` to the target computer and install the file by adding
the file to ``config.ini`` on the target computer.

You can simply load the folder on other computers by running python inside the folder and run

.. code-block:: python

    # import everything we need
    from astroNN.models import load_folder

    net = load_folder()

OR outside the folder `trained_models_folder`

.. code-block:: python

    # import everything we need
    from astroNN.models import load_folder

    net = load_folder("trained_models_folder")


NeuralNetMaster Class
--------------------------------------

NeuralNetMaster is the top level abstract class for all astroNN sub neural network classes. NeuralNetMaster define the
structure of how an astroNN neural network class should look like.

NeuralNetMaster consists of a pre-training checking (check input and labels shape, cpu/gpu check and create astroNN
folder for every run.

---------------------------------------------------------------
When `train()` is called from an astroNN neural net instance
---------------------------------------------------------------

When `train()` is called, the method will call `pre_training_checklist_child()` defined in the corresponding child class
and call `pre_training_checklist_master()` defined in `NeuralNetMaster`. `pre_training_checklist_master()` basically responsible
to do basic data checking, create an astroNN folder for this run and save hyperparameters.

After `pre_training_checklist_master()` has finished, `pre_training_checklist_child()` will run its checklist, including
normalizing data, compile model and setup the data generator which will yield data to the neural net during training.
