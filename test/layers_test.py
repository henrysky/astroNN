import unittest
import numpy as np
import numpy.testing as npt


from astroNN import keras_import_manager
keras = keras_import_manager()

Input = keras.layers.Input
Dense = keras.layers.Dense
Model = keras.models.Model


class LayerCase(unittest.TestCase):
    def test_BayesianDropout(self):
        print('==========BayesianDropout test==========')
        from astroNN.nn.layers import BayesianDropout

        # Data preparation
        random_xdata = np.random.normal(0, 1, (1000, 7514))
        random_ydata = np.random.normal(0, 1, (1000, 25))

        input = Input(shape=[7514])
        dense = Dense(100)(input)
        b_dropout = BayesianDropout(0.2)(dense)
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
        random_xdata = np.random.normal(0, 1, (1000, 7514))
        random_ydata = np.random.normal(0, 1, (1000, 25))

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


if __name__ == '__main__':
    unittest.main()
