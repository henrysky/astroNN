# ---------------------------------------------------------------#
#   astroNN.nn.metrics: metrics
# ---------------------------------------------------------------#
import tensorflow as tf

from astroNN.nn.losses import binary_accuracy, binary_accuracy_from_logits
from astroNN.nn.losses import categorical_accuracy
from astroNN.nn.losses import mean_absolute_error
from astroNN.nn.losses import mean_absolute_percentage_error
from astroNN.nn.losses import mean_error
from astroNN.nn.losses import mean_percentage_error
from astroNN.nn.losses import mean_squared_error
from astroNN.nn.losses import mean_squared_logarithmic_error
from astroNN.nn.losses import median
from astroNN.nn.losses import median_absolute_deviation
from astroNN.nn.losses import median_error
from astroNN.nn.losses import mad_std

# Just alias functions
mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
msle = mean_squared_logarithmic_error
me = mean_error
mpe = mean_percentage_error
categorical_accuracy = categorical_accuracy
binary_accuracy = binary_accuracy
binary_accuracy_from_logits = binary_accuracy_from_logits
mad = median_absolute_deviation
