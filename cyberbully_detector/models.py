from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import multi_gpu_model
from keras.callbacks import TerminateOnNaN
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import applications as kapps
import logging as log

def mse_nan(y_true, y_pred):
  # This loss function excludes NaN
  # Code taken from GitHub comment:
  # https://github.com/keras-team/keras/issues/9549
  index = ~K.tf.is_nan(y_true)
  y_true = K.tf.boolean_mask(y_true, index)
  y_pred = K.tf.boolean_mask(y_pred, index)
  return K.mean((y_true - y_pred) ** 2)

def init_emotic_model(config):
  pass 


CUSTOM_MODELS = {
    "emotic": init_emotic_model,
}

