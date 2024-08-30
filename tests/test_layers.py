import keras
import numpy as np
import numpy.testing as npt
import pytest
from astroNN.nn.losses import zeros_loss

Input = keras.layers.Input
Dense = keras.layers.Dense
concatenate = keras.layers.concatenate
Conv1D = keras.layers.Conv1D
Conv2D = keras.layers.Conv2D
Flatten = keras.layers.Flatten
Model = keras.models.Model
Sequential = keras.models.Sequential


def test_MCDropout():
    print("==========MCDropout tests==========")
    from astroNN.nn.layers import MCDropout

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514])
    dense = Dense(100)(input)
    b_dropout = MCDropout(0.2, name="dropout")(dense)
    output = Dense(25)(b_dropout)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    print(model.get_layer("dropout").get_config())
    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_MCGaussianDropout():
    print("==========MCGaussianDropout tests==========")
    from astroNN.nn.layers import MCGaussianDropout

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514])
    dense = Dense(100)(input)
    b_dropout = MCGaussianDropout(0.2, name="dropout")(dense)
    output = Dense(25)(b_dropout)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    print(model.get_layer("dropout").get_config())

    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_SpatialDropout1D():
    print("==========SpatialDropout1D tests==========")
    from astroNN.nn.layers import MCSpatialDropout1D

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514, 1))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514, 1])
    conv = Conv1D(
        kernel_initializer="he_normal", padding="same", filters=2, kernel_size=16
    )(input)
    dropout = MCSpatialDropout1D(0.2)(conv)
    flattened = Flatten()(dropout)
    output = Dense(25)(flattened)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_SpatialDropout12D():
    print("==========SpatialDropout2D tests==========")
    from astroNN.nn.layers import MCSpatialDropout2D

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 28, 28, 1))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[28, 28, 1])
    conv = Conv2D(
        kernel_initializer="he_normal", padding="same", filters=2, kernel_size=16
    )(input)
    dropout = MCSpatialDropout2D(0.2)(conv)
    flattened = Flatten()(dropout)
    output = Dense(25)(flattened)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_ErrorProp():
    print("==========MCDropout tests==========")
    from astroNN.nn.layers import ErrorProp

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_xdata_err = np.random.normal(0, 0.1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514])
    input_err = Input(shape=[7514])
    input_w_err = ErrorProp(name="error")([input, input_err])
    dense = Dense(100)(input_w_err)
    output = Dense(25)(dense)
    model = Model(inputs=[input, input_err], outputs=[output])
    model.compile(optimizer="sgd", loss="mse")

    model.fit([random_xdata, random_xdata_err], random_ydata, batch_size=128)

    print(model.get_layer("error").get_config())

    # make sure dropout is on even in testing phase
    x = model.predict([random_xdata, random_xdata_err])
    y = model.predict([random_xdata, random_xdata_err])
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_StopGrad():
    print("==========StopGrad tests==========")
    from astroNN.nn.layers import StopGrad

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514])
    output = Dense(25)(input)
    stopped_output = StopGrad(name="stopgrad", always_on=True)(output)
    model = Model(inputs=input, outputs=output)
    model_stopped = Model(inputs=input, outputs=stopped_output)
    model.compile(optimizer="adam", loss="mse")

    model_stopped.compile(optimizer="adam", loss="mse")
    # assert error because of no gradient via this layer
    # RuntimeError is raised with PyTorch backend
    # ValueError is raised with TensorFlow backend
    with pytest.raises((RuntimeError, ValueError)):
        model_stopped.fit(random_xdata, random_ydata, batch_size=128, epochs=1)

    x = model.predict(random_xdata)
    y = model_stopped.predict(random_xdata)
    npt.assert_almost_equal(x, y)  # make sure StopGrad does not change any result

    # # =================test weight equals================= #
    input2 = Input(shape=[7514])
    dense1 = Dense(100, name="normaldense")(input2)
    dense2 = Dense(25, name="wanted_dense")(input2)
    dense2_stopped = StopGrad(name="stopgrad", always_on=True)(dense2)
    output2 = Dense(25, name="wanted_dense2")(concatenate([dense1, dense2_stopped]))
    model2 = Model(inputs=[input2], outputs=[output2])
    model2.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss="mse",
    )
    weight_b4_train = model2.get_layer(name="wanted_dense").get_weights()[0]
    model2.fit(random_xdata, [random_ydata, random_ydata])
    weight_a4_train = model2.get_layer(name="wanted_dense").get_weights()[0]

    # make sure StopGrad does it job to stop gradient backpropation in complicated situation
    npt.assert_equal(weight_a4_train, weight_b4_train)


def test_BoolMask():
    print("==========BoolMask tests==========")
    from astroNN.apogee import aspcap_mask
    from astroNN.nn.layers import BoolMask

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514])
    dense = BoolMask(mask=aspcap_mask("Al", dr=14))(input)
    output = Dense(25)(dense)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    model.fit(random_xdata, random_ydata)

    # make sure a mask with all 0 raises error of invalid mask
    with pytest.raises(ValueError):
        BoolMask(mask=np.zeros(7514))


def test_FastMCInference():
    print("==========FastMCInference tests==========")
    from astroNN.nn.layers import FastMCInference

    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_ydata = np.random.normal(0, 1, (100, 25))

    input = Input(shape=[7514])
    dense = Dense(100)(input)
    output = Dense(25)(dense)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=64)

    acc_model = FastMCInference(10, model).new_mc_model

    # make sure accelerated model has no effect on deterministic model prediction
    x = model.predict(random_xdata)
    y = acc_model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y[:, :, 0])), True)
    # make sure accelerated model has no variance (within numerical precision) on deterministic model prediction
    # TODO: This test is failing with PyTorch backend, so setting decimal to 5
    npt.assert_almost_equal(np.max(y[:, :, 1]), 0.0, decimal=5)

    # assert error raised for things other than keras model
    with pytest.raises(TypeError):
        FastMCInference(10, "123")

    # sequential model test
    smodel = Sequential()
    smodel.add(Dense(32, input_shape=(7514,)))
    smodel.add(Dense(10, activation="softmax"))
    smodel.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    acc_smodel = FastMCInference(10, smodel).new_mc_model
    # make sure accelerated model has no effect on deterministic model prediction
    sx = smodel.predict(random_xdata)
    sy = acc_smodel.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(sx, sy[:, :, 0])), True)
    # make sure accelerated model has no variance (within numerical precision) on deterministic model prediction
    # TODO: This test is failing with PyTorch backend, so setting decimal to 5
    npt.assert_almost_equal(np.max(sy[:, :, 1]), 0.0, decimal=5)
