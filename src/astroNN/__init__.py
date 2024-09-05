from importlib.metadata import version

import keras

version = __version__ = version("astroNN")
_KERAS_BACKEND = keras.backend.backend()
_SUPPORTED_BACKEND = ["tensorflow", "torch"]

# check if the backend is compatible
if _KERAS_BACKEND not in _SUPPORTED_BACKEND:
    raise ImportError(
        f"astroNN only supports PyTorch or Tensorflow backend, but your current backend is {keras.backend.backend()}"
    )
