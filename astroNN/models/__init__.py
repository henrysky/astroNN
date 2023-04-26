import importlib
import json
import os
import sys
import warnings
from packaging import version

import h5py
from astroNN.config import custom_model_path_reader
from astroNN.models.apogee_models import (
    ApogeeBCNN,
    ApogeeCVAE,
    ApogeeCNN,
    ApogeeBCNNCensored,
    ApogeeDR14GaiaDR2BCNN,
    ApogeeKplerEchelle,
    ApogeeBCNNaux,
    ApokascEncoderDecoder,
    StarNet2017,
)
from astroNN.models.misc_models import Cifar10CNN, MNIST_BCNN, SimplePolyNN
from astroNN.nn.losses import losses_lookup
from astroNN.nn.utilities import Normalizer
from astroNN.shared.dict_tools import dict_list_to_dict_np, list_to_dict
from tensorflow import keras as tfk
import tensorflow as tf

__all__ = [
    "load_folder",
    "ApogeeBCNN",
    "ApogeeCVAE",
    "ApogeeCNN",
    "ApogeeBCNNCensored",
    "ApogeeDR14GaiaDR2BCNN",
    "ApogeeKplerEchelle",
    "ApogeeBCNNaux",
    "ApokascEncoderDecoder",
    "StarNet2017",
    "Cifar10CNN",
    "MNIST_BCNN",
    "SimplePolyNN",
]

optimizers = tfk.optimizers
Sequential = tfk.models.Sequential


def Galaxy10CNN():
    """
    NAME:
        Galaxy10CNN
    PURPOSE:
        setup Galaxy10CNN from Cifar10CNN with Galaxy10 parameter
    INPUT:
    OUTPUT:
        (instance): a callable instances from Cifar10_CNN with Galaxy10 parameter
    HISTORY:
        2018-Feb-09 - Written - Henry Leung (University of Toronto)
        2018-Apr-02 - Update - Henry Leung (University of Toronto)
    """
    from astroNN.datasets.galaxy10 import galaxy10cls_lookup
    from astroNN.models.misc_models import Cifar10CNN

    galaxy10_net = Cifar10CNN()
    galaxy10_net._model_identifier = "Galaxy10CNN"
    targetname = []
    for i in range(10):
        targetname.extend([galaxy10cls_lookup(i)])

    galaxy10_net.targetname = targetname
    return galaxy10_net


def convert_custom_objects(obj):
    """Handles custom object lookup.

    Based on Keras Source Code

    # Arguments
        obj: object, dict, or list.

    # Returns
        The same structure, where occurrences
            of a custom object name have been replaced
            with the custom object.
    """
    if isinstance(obj, list):
        deserialized = []
        for value in obj:
            deserialized.append(convert_custom_objects(value))
        return deserialized
    if isinstance(obj, dict):
        deserialized = {}
        for key, value in obj.items():
            deserialized[key] = convert_custom_objects(value)
        return deserialized
    return obj


def load_folder(folder=None):
    """
    To load astroNN model object from folder

    :param folder: [optional] you should provide folder name if outside folder, do not specific when you are inside the folder
    :type folder: str
    :return: astroNN Neural Network instance
    :rtype: astroNN.nn.NeuralNetMaster.NeuralNetMaster
    :History: 2017-Dec-29 - Written - Henry Leung (University of Toronto)
    """
    currentdir = os.getcwd()

    if folder is not None:
        fullfilepath = os.path.abspath(os.path.join(currentdir, folder))
    else:
        fullfilepath = os.path.abspath(currentdir)

    astronn_model_obj = None

    if (
        folder is not None
        and os.path.isfile(os.path.join(folder, "astroNN_model_parameter.json")) is True
    ):
        with open(os.path.join(folder, "astroNN_model_parameter.json")) as f:
            parameter = json.load(f)
            f.close()
    elif os.path.isfile("astroNN_model_parameter.json") is True:
        with open("astroNN_model_parameter.json") as f:
            parameter = json.load(f)
            f.close()
    elif folder is not None and not os.path.exists(folder):
        raise IOError("Folder not exists: " + str(currentdir + os.sep + folder))
    else:
        raise FileNotFoundError("Are you sure this is an astroNN generated folder?")

    identifier = parameter["id"]
    unknown_model_message = f"Unknown model identifier -> {identifier}!"

    # need to point to the actual neural network if non-travial location
    if identifier == "Galaxy10CNN":
        astronn_model_obj = Galaxy10CNN()
    else:
        # else try to import it from standard way
        try:
            astronn_model_obj = getattr(
                importlib.import_module(f"astroNN.models"), identifier
            )()
        except ImportError:
            # try to load custom model from CUSTOM_MODEL_PATH if none are working
            CUSTOM_MODEL_PATH = custom_model_path_reader()
            # try the current folder and see if there is any .py on top of CUSTOM_MODEL_PATH
            list_py_files = [
                os.path.join(fullfilepath, f)
                for f in os.listdir(fullfilepath)
                if f.endswith(".py")
            ]
            if CUSTOM_MODEL_PATH is None and list_py_files is None:
                print("\n")
                raise TypeError(unknown_model_message)
            else:
                for path_list in (
                    path_list
                    for path_list in [CUSTOM_MODEL_PATH, list_py_files]
                    if path_list is not None
                ):
                    for path in path_list:
                        head, tail = os.path.split(path)
                        sys.path.insert(0, head)
                        try:
                            model = getattr(
                                importlib.import_module(tail.strip(".py")),
                                str(identifier),
                            )
                            astronn_model_obj = model()
                        except AttributeError:
                            pass

        if astronn_model_obj is None:
            print("\n")
            raise TypeError(unknown_model_message)

    astronn_model_obj.currentdir = currentdir
    astronn_model_obj.fullfilepath = fullfilepath
    astronn_model_obj.folder_name = (
        folder if folder is not None else os.path.basename(os.path.normpath(currentdir))
    )

    # Must have parameter
    astronn_model_obj._input_shape = parameter["input"]
    astronn_model_obj._labels_shape = parameter["labels"]
    if type(astronn_model_obj._input_shape) is not dict:
        astronn_model_obj._input_shape = {"input": astronn_model_obj._input_shape}
    if type(astronn_model_obj._labels_shape) is not dict:
        astronn_model_obj._labels_shape = {"output": astronn_model_obj._labels_shape}
    astronn_model_obj.num_hidden = parameter["hidden"]
    astronn_model_obj.input_norm_mode = parameter["input_norm_mode"]
    astronn_model_obj.labels_norm_mode = parameter["labels_norm_mode"]
    astronn_model_obj.batch_size = parameter["batch_size"]
    astronn_model_obj.targetname = parameter["targetname"]
    astronn_model_obj.val_size = parameter["valsize"]

    # Conditional parameter depends on neural net architecture
    try:
        astronn_model_obj.num_filters = parameter["filternum"]
    except KeyError:
        pass
    try:
        astronn_model_obj.filter_len = parameter["filterlen"]
    except KeyError:
        pass
    try:
        pool_length = parameter["pool_length"]
        if pool_length is not None:
            if isinstance(pool_length, int):  # multi-dimensional case
                astronn_model_obj.pool_length = parameter["pool_length"]
            else:
                astronn_model_obj.pool_length = list(parameter["pool_length"])
    except KeyError or TypeError:
        pass
    try:
        # need to convert to int because of keras do not want array or list
        astronn_model_obj.latent_dim = int(parameter["latent"])
    except KeyError:
        pass
    try:
        astronn_model_obj.task = parameter["task"]
    except KeyError:
        pass
    try:
        astronn_model_obj.dropout_rate = parameter["dropout_rate"]
    except KeyError:
        pass
    try:
        # if inverse model precision exists, so does length_scale
        astronn_model_obj.inv_model_precision = parameter["inv_tau"]
        astronn_model_obj.length_scale = parameter["length_scale"]
    except KeyError:
        pass
    try:
        astronn_model_obj.l1 = parameter["l1"]
    except KeyError:
        pass
    try:
        astronn_model_obj.l2 = parameter["l2"]
    except KeyError:
        pass
    try:
        astronn_model_obj.maxnorm = parameter["maxnorm"]
    except KeyError:
        pass
    try:
        astronn_model_obj.input_names = parameter["input_names"]
    except KeyError:
        astronn_model_obj.input_names = ["input"]
    try:
        astronn_model_obj.output_names = parameter["output_names"]
    except KeyError:
        astronn_model_obj.output_names = ["output"]
    try:
        astronn_model_obj._last_layer_activation = parameter["last_layer_activation"]
    except KeyError:
        pass
    try:
        astronn_model_obj.activation = parameter["activation"]
    except KeyError:
        pass
    try:
        astronn_model_obj.aux_length = parameter["aux_length"]
    except KeyError:
        pass
    with h5py.File(
        os.path.join(astronn_model_obj.fullfilepath, "model_weights.h5"), mode="r"
    ) as f:
        training_config = json.loads(f.attrs["training_config"])
        optimizer_config = training_config["optimizer_config"]
        optimizer = optimizers.deserialize(optimizer_config)
        model_config = json.loads(f.attrs["model_config"])
        # for older models, they have -tf prefix like 2.1.6-tf which cannot be parsed by version
        tfk_version = (f.attrs["keras_version"]).replace("-tf", "")

        # input/name names, mean, std
        input_names = []
        output_names = []
        for lay in model_config["config"]["input_layers"]:
            input_names.append(lay[0])
        for lay in model_config["config"]["output_layers"]:
            output_names.append(lay[0])
        astronn_model_obj.input_mean = list_to_dict(
            input_names, dict_list_to_dict_np(parameter["input_mean"])
        )
        astronn_model_obj.labels_mean = list_to_dict(
            output_names, dict_list_to_dict_np(parameter["labels_mean"])
        )
        astronn_model_obj.input_std = list_to_dict(
            input_names, dict_list_to_dict_np(parameter["input_std"])
        )
        astronn_model_obj.labels_std = list_to_dict(
            output_names, dict_list_to_dict_np(parameter["labels_std"])
        )

        # Recover loss functions and metrics.
        losses_raw = convert_custom_objects(training_config["loss"])
        if losses_raw:
            try:
                try:
                    loss = [losses_lookup(losses_raw[_loss]) for _loss in losses_raw]
                except TypeError:
                    loss = losses_lookup(losses_raw)
            except:
                raise LookupError("Cant lookup loss")
        else:
            loss = None

        metrics_raw = convert_custom_objects(training_config["metrics"])
        # its weird that keras needs -> metrics[metric][0] instead of metrics[metric] likes losses
        try:
            try:
                if version.parse(tf.__version__) >= version.parse("2.4.0"):
                    metrics = [
                        losses_lookup(_metric["config"]["fn"])
                        for _metric in metrics_raw[0]
                    ]
                else:
                    metrics = [
                        losses_lookup(metrics_raw[_metric][0])
                        for _metric in metrics_raw
                    ]
            except TypeError:
                metrics = [losses_lookup(metrics_raw[0])]
        except:
            metrics = metrics_raw

        sample_weight_mode = (
            training_config["sample_weight_mode"]
            if hasattr(training_config, "sample_weight_mode")
            else None
        )
        loss_weights = training_config["loss_weights"]
        weighted_metrics = None

        # compile the model
        astronn_model_obj.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            loss_weights=loss_weights,
            sample_weight_mode=sample_weight_mode,
        )

        # set weights
        astronn_model_obj.keras_model.load_weights(
            os.path.join(astronn_model_obj.fullfilepath, "model_weights.h5")
        )

        # Build train function (to get weight updates), need to consider Sequential model too
        astronn_model_obj.keras_model.make_train_function()
        try:
            optimizer_weights_group = f["optimizer_weights"]
            if version.parse(h5py.__version__) >= version.parse("3.0"):
                optimizer_weight_names = [
                    n for n in optimizer_weights_group.attrs["weight_names"]
                ]
            else:
                optimizer_weight_names = [
                    n.decode("utf8")
                    for n in optimizer_weights_group.attrs["weight_names"]
                ]
            optimizer_weight_values = [
                optimizer_weights_group[n] for n in optimizer_weight_names
            ]
            # TODO: switch to new optimzer API after we have dropped tf2.10 support
            if version.parse(tfk_version) > version.parse("2.10.99"):
                astronn_model_obj.keras_model.optimizer.build(
                    astronn_model_obj.keras_model.trainable_variables
                )
            else:
                astronn_model_obj.keras_model.optimizer._create_all_weights(
                    astronn_model_obj.keras_model.trainable_variables
                )
            astronn_model_obj.keras_model.optimizer.set_weights(optimizer_weight_values)
        except KeyError:
            warnings.warn(
                "Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer."
            )

    # create normalizer and set correct mean and std
    astronn_model_obj.input_normalizer = Normalizer(
        mode=astronn_model_obj.input_norm_mode
    )
    astronn_model_obj.labels_normalizer = Normalizer(
        mode=astronn_model_obj.labels_norm_mode
    )
    astronn_model_obj.input_normalizer.mean_labels = astronn_model_obj.input_mean
    astronn_model_obj.input_normalizer.std_labels = astronn_model_obj.input_std
    astronn_model_obj.labels_normalizer.mean_labels = astronn_model_obj.labels_mean
    astronn_model_obj.labels_normalizer.std_labels = astronn_model_obj.labels_std
    print("========================================================")
    print(f"Loaded astroNN model, model type: {astronn_model_obj.name} -> {identifier}")
    print("========================================================")
    return astronn_model_obj
