import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from tensorflow import keras as tfk

from astroNN.nn.losses import zeros_loss
from astroNN.shared.nn_tools import gpu_memory_manage

Input = tfk.layers.Input
Dense = tfk.layers.Dense
concatenate = tfk.layers.concatenate
Conv1D = tfk.layers.Conv1D
Conv2D = tfk.layers.Conv2D
Flatten = tfk.layers.Flatten
Model = tfk.models.Model
Sequential = tfk.models.Sequential

gpu_memory_manage()


class LayerCase(unittest.TestCase):
    def test_MCDropout(self):
        print('==========MCDropout tests==========')
        from astroNN.nn.layers import MCDropout

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = Dense(100)(input)
        b_dropout = MCDropout(0.2, name='dropout')(dense)
        output = Dense(25)(b_dropout)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        print(model.get_layer('dropout').get_config())
        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        self.assertEqual(np.any(np.not_equal(x, y)), True)

    def test_MCGaussianDropout(self):
        print('==========MCGaussianDropout tests==========')
        from astroNN.nn.layers import MCGaussianDropout

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = Dense(100)(input)
        b_dropout = MCGaussianDropout(0.2, name='dropout')(dense)
        output = Dense(25)(b_dropout)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        print(model.get_layer('dropout').get_config())

        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        self.assertEqual(np.any(np.not_equal(x, y)), True)

    def test_ConcreteDropout(self):
        print('==========ConcreteDropout tests==========')
        from astroNN.nn.layers import MCConcreteDropout

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = MCConcreteDropout(Dense(100), name='dropout')(input)
        output = Dense(25)(dense)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        print(model.get_layer('dropout').get_config())

        # make sure dropout is on even in testing phase
        x = model.predict(random_xdata)
        y = model.predict(random_xdata)
        self.assertEqual(np.any(np.not_equal(x, y)), True)

    def test_SpatialDropout1D(self):
        print('==========SpatialDropout1D tests==========')
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
        self.assertEqual(np.any(np.not_equal(x, y)), True)

    def test_SpatialDropout12D(self):
        print('==========SpatialDropout2D tests==========')
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
        self.assertEqual(np.any(np.not_equal(x, y)), True)

    def test_ErrorProp(self):
        print('==========MCDropout tests==========')
        from astroNN.nn.layers import ErrorProp

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_xdata_err = np.random.normal(0, 0.1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        input_err = Input(shape=[7514])
        input_w_err = ErrorProp(name='error')([input, input_err])
        dense = Dense(100)(input_w_err)
        output = Dense(25)(dense)
        model = Model(inputs=[input, input_err], outputs=[output])
        model.compile(optimizer='sgd', loss='mse')

        model.fit([random_xdata, random_xdata_err], random_ydata, batch_size=128)

        print(model.get_layer('error').get_config())

        # make sure dropout is on even in testing phase
        x = model.predict([random_xdata, random_xdata_err])
        y = model.predict([random_xdata, random_xdata_err])
        self.assertEqual(np.any(np.not_equal(x, y)), True)

    # def test_MCBN(self):
    #     print('==========MCDropout tests==========')
    #     from astroNN.nn.layers import MCBatchNorm
    #
    #     # Data preparation
    #     random_xdata = np.random.normal(0, 1, (100, 7514))
    #     random_ydata = np.random.normal(0, 1, (100, 25))
    #
    #     input = Input(shape=[7514])
    #     dense = Dense(100)(input)
    #     b_dropout = MCBatchNorm(name='MCBN')(dense)
    #     output = Dense(25)(b_dropout)
    #     model = Model(inputs=input, outputs=output)
    #     model.compile(optimizer='sgd', loss='mse')
    #
    #     model.fit(random_xdata, random_ydata, batch_size=128)
    #
    #     print(model.get_layer('MCBN').get_config())
    #
    #     # make sure dropout is on even in testing phase
    #     x = model.predict(random_xdata)
    #     y = model.predict(random_xdata)
    #     self.assertEqual(np.all(np.not_equal(x, y)), False)

    def test_StopGrad(self):
        print('==========StopGrad tests==========')
        from astroNN.nn.layers import StopGrad

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        output = Dense(25)(input)
        stopped_output = StopGrad(name='stopgrad', always_on=True)(output)
        model = Model(inputs=input, outputs=output)
        model_stopped = Model(inputs=input, outputs=stopped_output)
        model.compile(optimizer='adam', loss='mse')

        model_stopped.compile(optimizer='adam', loss='mse')
        # assert error because of no gradient via this layer
        self.assertRaises(ValueError, model_stopped.fit, random_xdata, random_ydata, batch_size=128, epochs=1)
        # keras.backend.get_session().run(keras.backend.gradients(output, input), feed_dict={input: random_xdata, keras.backend.learning_phase(): 0})

        x = model.predict(random_xdata)
        y = model_stopped.predict(random_xdata)
        npt.assert_almost_equal(x, y)  # make sure StopGrad does not change any result

        # # =================test weight equals================= #
        input2 = Input(shape=[7514])
        dense1 = Dense(100, name='normaldense')(input2)
        dense2 = Dense(25, name='wanted_dense')(input2)
        dense2_stopped = StopGrad(name='stopgrad', always_on=True)(dense2)
        output2 = Dense(25, name='wanted_dense2')(concatenate([dense1, dense2_stopped]))
        model2 = Model(inputs=input2, outputs=[output2, dense2])
        model2.compile(optimizer=tfk.optimizers.SGD(lr=0.1),
                       loss={'wanted_dense2': 'mse', 'wanted_dense': zeros_loss})
        weight_b4_train = model2.get_layer(name='wanted_dense').get_weights()[0]
        model2.fit(random_xdata, [random_ydata, random_ydata])
        weight_a4_train = model2.get_layer(name='wanted_dense').get_weights()[0]

        # make sure StopGrad does it job to stop gradient backpropation in complicated situation
        self.assertEqual(np.all(weight_a4_train == weight_b4_train), True)

    def test_BoolMask(self):
        print('==========BoolMask tests==========')
        from astroNN.nn.layers import BoolMask
        from astroNN.apogee import aspcap_mask

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = BoolMask(mask=aspcap_mask("Al", dr=14))(input)
        output = Dense(25)(dense)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        model.fit(random_xdata, random_ydata)

        # make sure a mask with all 0 raises error of invalid mask
        self.assertRaises(ValueError, BoolMask, np.zeros(7514))

    def test_FastMCInference(self):
        print('==========FastMCInference tests==========')
        from astroNN.nn.layers import FastMCInference

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        input = Input(shape=[7514])
        dense = Dense(100)(input)
        output = Dense(25)(dense)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=128)

        acc_model = FastMCInference(10)(model)

        # make sure accelerated model has no effect on deterministic model prediction
        x = model.predict(random_xdata)
        y = acc_model.predict(random_xdata)
        self.assertEqual(np.any(np.not_equal(x, y[:, :, 0])), True)
        # make sure accelerated model has no variance (uncertainty) on deterministic model prediction
        self.assertAlmostEqual(np.sum(y[:, :, 1]), 0.)

        # assert error raised for things other than keras model
        self.assertRaises(TypeError, FastMCInference(10), '123')

        # sequential model test
        smodel = Sequential()
        smodel.add(Dense(32, input_shape=(7514,)))
        smodel.add(Dense(10, activation='softmax'))
        smodel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        acc_smodel = FastMCInference(10)(smodel)
        # make sure accelerated model has no effect on deterministic model prediction
        sx = smodel.predict(random_xdata)
        sy = acc_smodel.predict(random_xdata)
        self.assertEqual(np.any(np.not_equal(sx, sy[:, :, 0])), True)
        # make sure accelerated model has no variance (uncertainty) on deterministic model prediction
        self.assertAlmostEqual(np.sum(sy[:, :, 1]), 0.)

    def test_TensorInput(self):
        print('==========BoolMask tests==========')
        from astroNN.nn.layers import TensorInput

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))

        # Data preparation
        random_xdata = np.random.normal(0, 1, (100, 7514))
        random_ydata = np.random.normal(0, 1, (100, 25))
        input1 = Input(shape=[7514], name='input')
        input2 = TensorInput(tensor=tf.random.normal(mean=0., stddev=1., shape=tf.shape(input1)))([])
        output = Dense(25, name='dense')(concatenate([input1, input2]))
        model = Model(inputs=input1, outputs=output)
        model.compile(optimizer='adam', loss='mse')

        self.assertEqual(len(model.input_names), 1)

    def test_PolyFit(self):
        print('==========PolyFit tests==========')
        from astroNN.nn.layers import PolyFit

        # Data preparation
        polynomial_coefficient = [0.1, -0.05]
        random_xdata = np.random.normal(0, 3, (100, 1))
        random_ydata = polynomial_coefficient[1] * random_xdata + polynomial_coefficient[0]

        self.assertRaises(ValueError, PolyFit, deg=1, use_xbias=False, init_w=[2., 3., 4.])

        input = Input(shape=[1, ])
        output = PolyFit(deg=1, use_xbias=False, init_w=[[[0.1]], [[-0.05]]], name='polyfit')(input)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=32, epochs=1)
        # no gradient update thus the answer should equal to the real equation anyway
        self.assertEqual(np.allclose(model.predict(random_xdata), random_ydata), True)
        # no gradients update because initial weights are perfect
        npt.assert_almost_equal(np.squeeze(model.get_weights()[0]), polynomial_coefficient)
        print(model.get_layer('polyfit').get_config())

        # Data preparation
        polynomial_coefficient = [0.075, -0.05]
        random_xdata = np.random.normal(0, 1, (100, 2))
        random_ydata = np.vstack((np.sum(polynomial_coefficient[1] * random_xdata + polynomial_coefficient[0], axis=1),
                                  np.sum(polynomial_coefficient[1] * random_xdata + polynomial_coefficient[0],
                                         axis=1))).T

        input = Input(shape=[2, ])
        output = PolyFit(deg=1, output_units=2, use_xbias=False,
                         init_w=[[[0.075, 0.075], [0.075, 0.075]], [[-0.05, -0.05], [-0.05, -0.05]]], name='PolyFit')(
            input)
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer='sgd', loss='mse')

        model.fit(random_xdata, random_ydata, batch_size=32, epochs=1)
        npt.assert_almost_equal(model.predict(random_xdata), random_ydata)

        # no gradients update because initial weights are perfect
        npt.assert_almost_equal(np.squeeze(model.get_weights()[0]),
                                [[[0.075, 0.075], [0.075, 0.075]], [[-0.05, -0.05], [-0.05, -0.05]]])

    # def test_BayesPolyFit(self):
    #     # this layer currently cannot be run with keras duh!
    #     print('==========BayesPolyFit tests==========')
    #     from astroNN.nn.layers import BayesPolyFit
    #
    #     # Data preparation
    #     polynomial_coefficient = [0.1, -0.05]
    #     random_xdata = np.random.normal(0, 2, (100, 1))
    #     random_ydata = polynomial_coefficient[1] * random_xdata + polynomial_coefficient[0] + \
    #                    np.random.normal(0, 0.01, (100, 1))
    #
    #     self.assertRaises(ValueError, BayesPolyFit, deg=3, use_xbias=False, init_w=[2., 3., 4.])
    #
    #     input = Input(shape=[1, ])
    #     output = BayesPolyFit(deg=1, use_xbias=True, name='polyfit')(input)
    #     model = Model(inputs=input, outputs=output)
    #     model.compile(optimizer='sgd', loss='mse')
    #     model.fit(random_xdata, random_ydata, batch_size=32, epochs=1)
    #     print("Mean: ", model.get_layer("polyfit").get_weights_and_error()['weights'])
    #     print("STD: ", model.get_layer("polyfit").get_weights_and_error()['error'])
    #     print(model.losses[0].eval(session=tfk.backend.get_session()))


if __name__ == '__main__':
    unittest.main()
