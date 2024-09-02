import keras
import numpy as np
import numpy.testing as npt
import pytest
import astroNN.nn.layers
from astroNN.apogee import aspcap_mask


@pytest.fixture(scope="module")
def random_data():
    # Data preparation
    random_xdata = np.random.normal(0, 1, (100, 7514))
    random_xdata_err = np.abs(np.random.normal(0, 0.1, (100, 7514)))
    random_ydata = np.random.normal(0, 1, (100, 25))
    random_ydata_err = np.abs(np.random.normal(0, 0.1, (100, 7514)))
    return random_xdata, random_xdata_err, random_ydata, random_ydata_err


def test_MCDropout(random_data):
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    input = keras.layers.Input(shape=[7514])
    dense = keras.layers.Dense(100)(input)
    b_dropout = astroNN.nn.layers.MCDropout(0.2, name="dropout")(dense)
    output = keras.layers.Dense(25)(b_dropout)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    print(model.get_layer("dropout").get_config())
    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_MCGaussianDropout(random_data):
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    input = keras.layers.Input(shape=[7514])
    dense = keras.layers.Dense(100)(input)
    b_dropout = astroNN.nn.layers.MCGaussianDropout(0.2, name="dropout")(dense)
    output = keras.layers.Dense(25)(b_dropout)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    print(model.get_layer("dropout").get_config())

    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_SpatialDropout1D(random_data):
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    input = keras.layers.Input(shape=[7514, 1])
    conv = keras.layers.Conv1D(
        kernel_initializer="he_normal", padding="same", filters=2, kernel_size=16
    )(input)
    dropout = astroNN.nn.layers.MCSpatialDropout1D(0.2)(conv)
    flattened = keras.layers.Flatten()(dropout)
    output = keras.layers.Dense(25)(flattened)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(random_xdata, random_ydata, batch_size=128)

    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_SpatialDropout12D(mnist_data):
    random_xdata, random_ydata, _, _ = mnist_data

    input = keras.layers.Input(shape=[28, 28, 1])
    conv = keras.layers.Conv2D(
        kernel_initializer="he_normal", padding="same", filters=2, kernel_size=16
    )(input)
    dropout = astroNN.nn.layers.MCSpatialDropout2D(0.2)(conv)
    flattened = keras.layers.Flatten()(dropout)
    output = keras.layers.Dense(10, activation="softmax")(flattened)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    model.fit(random_xdata, random_ydata)

    # make sure dropout is on even in testing phase
    x = model.predict(random_xdata)
    y = model.predict(random_xdata)
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_ErrorProp(random_data):
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    input = keras.layers.Input(shape=[7514])
    input_err = keras.layers.Input(shape=[7514])
    input_w_err = astroNN.nn.layers.ErrorProp(name="error")([input, input_err])
    dense = keras.layers.Dense(100)(input_w_err)
    output = keras.layers.Dense(25)(dense)
    model = keras.models.Model(inputs=[input, input_err], outputs=[output])
    model.compile(optimizer="sgd", loss="mse")

    model.fit([random_xdata, random_xdata_err], random_ydata, batch_size=128)

    print(model.get_layer("error").get_config())

    # make sure dropout is on even in testing phase
    x = model.predict([random_xdata, random_xdata_err])
    y = model.predict([random_xdata, random_xdata_err])
    npt.assert_equal(np.any(np.not_equal(x, y)), True)


def test_StopGrad(random_data):
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    input = keras.layers.Input(shape=[7514])
    output = keras.layers.Dense(25)(input)
    stopped_output = astroNN.nn.layers.StopGrad(name="stopgrad", always_on=True)(output)
    model = keras.models.Model(inputs=input, outputs=output)
    model_stopped = keras.models.Model(inputs=input, outputs=stopped_output)
    model.compile(optimizer="adam", loss="mse")
    model_stopped.compile(optimizer="adam", loss="mse")
    # assert error because of no gradient via this layer
    # RuntimeError is raised with PyTorch backend
    # ValueError is raised with TensorFlow backend
    with pytest.raises((RuntimeError, ValueError)):
        model_stopped.fit(random_xdata, random_ydata)

    # make sure StopGrad does not change any result when predicting
    npt.assert_almost_equal(
        model.predict(random_xdata), model_stopped.predict(random_xdata), err_msg="StopGrad layer should not change result when predicting"
    )

    # =================test weight equals================= #
    input = keras.layers.Input(shape=[7514])
    dense1 = keras.layers.Dense(100, name="normal_dense")(input)
    dense2 = keras.layers.Dense(32, name="wanted_dense")(input)
    dense2_stopped = astroNN.nn.layers.StopGrad(name="stopgrad", always_on=True)(dense2)
    output = keras.layers.Dense(25, name="wanted_dense2")(
        keras.layers.concatenate([dense1, dense2_stopped])
    )
    model2 = keras.models.Model(inputs=[input], outputs=[output])
    model2.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss="mse",
    )
    weight_b4_train = model2.get_layer(name="wanted_dense").get_weights()[0]
    model2.fit(random_xdata, [random_ydata, random_ydata])
    weight_a4_train = model2.get_layer(name="wanted_dense").get_weights()[0]

    # make sure StopGrad does it job to stop gradient backpropation in complicated situation
    npt.assert_equal(weight_a4_train, weight_b4_train)


def test_BoolMask(random_data):
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    input = keras.layers.Input(shape=[7514])
    dense = astroNN.nn.layers.BoolMask(mask=aspcap_mask("Al", dr=14))(input)
    output = keras.layers.Dense(25)(dense)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    model.fit(random_xdata, random_ydata)

    # make sure a mask with all 0 raises error of invalid mask
    with pytest.raises(ValueError):
        astroNN.nn.layers.BoolMask(mask=np.zeros(7514))


def test_FastMCInference(random_data):
    """
    Test the FastMCInference layer

    We need to make sure the layer works in various Keras model types
    """
    random_xdata, random_xdata_err, random_ydata, random_ydata_err = random_data

    # ======== Simple Keras functional Model, one input one output ======== #
    input = keras.layers.Input(shape=[7514])
    dense = keras.layers.Dense(100)(input)
    output = keras.layers.Dense(25)(dense)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
    original_weights = model.get_weights()
    acc_model = astroNN.nn.layers.FastMCInference(10, model).transformed_model
    # make sure accelerated model has no effect on deterministic model weights
    npt.assert_equal(
        original_weights,
        acc_model.get_weights(),
        err_msg="FastMCInference layer should not change weights",
    )
    x = acc_model.predict(random_xdata)
    # make sure the shape is correct, 100 samples, 25 outputs, 2 columns (mean and variance)
    npt.assert_equal(
        x.shape,
        (100, 25, 2),
        err_msg="FastMCInference layer should return 2 columns in the last axis (mean and variance)",
    )
    # make sure accelerated model has no variance (within numerical precision) on deterministic model prediction
    npt.assert_almost_equal(
        np.max(x[:, :, 1]),
        0.0,
        err_msg="FastMCInference layer should return 0 variance for deterministic model",
    )

    # ======== Simple Keras sequential Model, one input one output ======== #
    smodel = keras.models.Sequential()
    smodel.add(keras.layers.Input(shape=(7514,)))
    smodel.add(keras.layers.Dense(32))
    smodel.add(keras.layers.Dense(10, activation="softmax"))
    smodel.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    acc_smodel = astroNN.nn.layers.FastMCInference(10, smodel).transformed_model
    x = acc_smodel.predict(random_xdata)
    # make sure the shape is correct, 100 samples, 10 outputs, 2 columns (mean and variance)
    npt.assert_equal(
        x.shape,
        (100, 10, 2),
        err_msg="FastMCInference layer should return 2 columns in the last axis (mean and variance)",
    )
    # make sure accelerated model has no variance (within numerical precision) on deterministic model prediction
    npt.assert_almost_equal(
        np.max(x[:, :, 1]),
        0.0,
        err_msg="FastMCInference layer should return 0 variance for deterministic model",
    )

    # ======== Complex Keras functional Model, one input multiple output ======== #
    input = keras.layers.Input(shape=[7514])
    dense = keras.layers.Dense(100)(input)
    output1 = keras.layers.Dense(4, name="output1")(dense)
    output2 = keras.layers.Dense(8, name="output2")(dense)
    model = keras.models.Model(
        inputs=input, outputs={"output1": output1, "output2": output2}
    )
    model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
    acc_model = astroNN.nn.layers.FastMCInference(10, model).transformed_model
    x = acc_model.predict(random_xdata)
    # make sure the shape is correct
    assert isinstance(
        x, dict
    ), "Output from FastMCInference layer should be a dictionary if model has multiple outputs"
    npt.assert_equal(
        x["output1"].shape,
        (100, 4, 2),
        err_msg="FastMCInference layer return errorous shape for model with multiple outputs",
    )
    npt.assert_equal(
        x["output2"].shape,
        (100, 8, 2),
        err_msg="FastMCInference layer return errorous shape for model with multiple outputs",
    )

    # ======== Simple Keras functional Model with randomness ======== #
    input = keras.layers.Input(shape=[7514])
    dense = keras.layers.Dense(100)(input)
    dropout = astroNN.nn.layers.MCDropout(0.5)(dense)
    output = keras.layers.Dense(25)(dropout)
    model = keras.models.Model(inputs=input, outputs=output)
    model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
    original_weights = model.get_weights()
    acc_model = astroNN.nn.layers.FastMCInference(10, model).transformed_model
    # make sure accelerated model has no effect on sochastic model weights
    npt.assert_equal(
        original_weights,
        acc_model.get_weights(),
        err_msg="FastMCInference layer should not change weights",
    )
    x = acc_model.predict(random_xdata)
    # make sure the shape is correct, 100 samples, 25 outputs, 2 columns (mean and variance)
    npt.assert_equal(
        x.shape,
        (100, 25, 2),
        err_msg="FastMCInference layer should return 2 columns in the last axis (mean and variance)",
    )
    # make sure accelerated model has variance because of dropout
    assert (
        np.median(x[:, :, 1]) > 1.0
    ), "FastMCInference layer should return some degree of variances for stochastic model"

    # assert error raised for things other than keras model
    with pytest.raises(TypeError):
        astroNN.nn.layers.FastMCInference(10, "123")
