# ---------------------------------------------------------#
#   astroNN.NN.model_eval: Evaluate CNN model
# ---------------------------------------------------------#

import random
import pylab as plt
from keras import backend as K
from keras.models import load_model
import h5py
import numpy as np
from functools import reduce
import os