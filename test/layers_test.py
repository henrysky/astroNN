import unittest
import numpy as np
import numpy.testing as npt

from astroNN.shared.nn_tools import gpu_memory_manage
from astroNN import keras_import_manager
keras = keras_import_manager()

Input = keras.layers.Input
Dense = keras.layers.Dense
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Flatten = keras.layers.Flatten
Model = keras.models.Model

gpu_memory_manage()


class LayerCase(unittest.TestCase):
    def test_MCDropout(self):
        print('==========MCDropout test==========')
        from astroNN.nn.layers import MCDropout

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = Dense(100)(input)
        b_dropout = MCDropout(0.2)(dense)
        output = Dense(25)(b_dropout)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        npt.assert_equal(np.any(np.not_equal(x, y)), True)

    def test_ConcreteDropout(self):
        print('==========ConcreteDropout test==========')
        from astroNN.nn.layers import ConcreteDropout

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = ConcreteDropout(Dense(100))(input)
        output = Dense(25)(dense)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        npt.assert_equal(np.any(np.not_equal(x, y)), True)

    def test_SpatialDropout1D(self):
        print('==========SpatialDropout1D test==========')
        from astroNN.nn.layers import MCSpatialDropout1D

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514, 1))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514, 1])
        conv = Conv1D(kernel_initializer='he_normal', padding="same", filters=2, kernel_size=16)(input)
        dropout = MCSpatialDropout1D(0.2)(conv)
        flattened = Flatten()(dropout)
        output = Dense(25)(flattened)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        npt.assert_equal(np.any(np.not_equal(x, y)), True)

    def test_SpatialDropout12D(self):
        print('==========SpatialDropout2D test==========')
        from astroNN.nn.layers import MCSpatialDropout2D

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 28, 28, 1))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[28, 28, 1])
        conv = Conv2D(kernel_initializer='he_normal', padding="same", filters=2, kernel_size=16)(input)
        dropout = MCSpatialDropout2D(0.2)(conv)
        flattened = Flatten()(dropout)
        output = Dense(25)(flattened)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        npt.assert_equal(np.any(np.not_equal(x, y)), True)


if __name__ == '__main__':
    unittest.main()
