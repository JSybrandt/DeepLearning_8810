import keras.backend as K

def mse_nan(y_true, y_pred):
  # This loss function excludes NaN
  # Code taken from GitHub comment:
  # https://github.com/keras-team/keras/issues/9549
  index = ~K.tf.is_nan(y_true)
  y_true = K.tf.boolean_mask(y_true, index)
  y_pred = K.tf.boolean_mask(y_pred, index)
  # Need max in case of nan
  return K.maximum(K.mean((y_true - y_pred) ** 2), K.zeros(shape=(1,)))
