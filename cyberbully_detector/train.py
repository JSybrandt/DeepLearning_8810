import logging as log
from .data_util import colormode_to_str
from .data_util import colormode_to_dim
from .data_util import setup_training_data_generator
from .data_util import get_worker_count
from .data_util import get_config
from .bully_pb2 import Config
from .proto_util import get_or_none
from .generator import ImageAndAnnotationGenerator
from . import labels_pb2 as LB
from .proto_util import enum_size
from .proto_util import simple_proto_size
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.utils import multi_gpu_model
from keras.callbacks import TerminateOnNaN
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import applications as kapps
import keras.backend as K
import numpy as np

def add_convolutional(conv_config, current_layer):
  name=get_or_none(conv_config, "name")
  current_layer = Conv2D(
      conv_config.filters,
      (conv_config.kernel_size.width, conv_config.kernel_size.height),
      name=name
  )(current_layer)

  if name is not None:
    name += "_pool"

  if conv_config.pool_type == "max":
    current_layer = MaxPooling2D(
        (conv_config.pool_size.width, conv_config.pool_size.height),
        name=name
    )(current_layer)
  else:
    raise ValueError("Unsupported pool type:" + conv_config.pool_type)
  return current_layer

def load_transfer_layers(layer_conf, current_layer):

  if not layer_conf.HasField("name"):
    raise ValueError("Transfer layer is required to have a name")

  if layer_conf.name.lower() == "xception":
    select_model = kapps.Xception
  elif layer_conf.name.lower() == "vgg16":
    select_model = kapps.VGG16
  elif layer_conf.name.lower() == "vgg19":
    select_model = kapps.VGG19
  elif layer_conf.name.lower() == "resnet50":
    select_model = kapps.ResNet50
  elif layer_conf.name.lower() == "inceptionv3":
    select_model = kapps.InceptionV3
  elif layer_conf.name.lower() == "inceptionresnetv2":
    select_model = kapps.InceptionResNetV2
  elif layer_conf.name.lower() == "mobilenet":
    select_model = kapps.MobileNet
  elif layer_conf.name.lower() == "densenet":
    select_model = kapps.DenseNet
  elif layer_conf.name.lower() == "nasnet":
    select_model = kapps.NASNet
  elif layer_conf.name.lower() == "mobilenetv2":
    select_model = kapps.MobileNetV2
  else:
    raise ValueError("Transfer layer contains invalid application name")

  # include_top refers to the post-flatten features
  transfer_model = select_model(include_top=False,
                             weights=layer_conf.transfer.weights,
                             input_tensor=current_layer)
  if layer_conf.transfer.freeze:
    for layer in transfer_model.layers:
      layer.trainable = False
  elif layer_conf.transfer.HasField("partial_freeze"):
    for layer in transfer_model.layers[:layer_conf.transfer.partial_freeze]:
      layer.trainable = False

  return transfer_model.output

def get_output_layers(num_people, prev_layer):
  outputs = []
  # Contains bullying
  outputs.append(Dense(1,
                      activation="sigmoid",
                      name="contains_bullying")(prev_layer))
  # Bully Type
  outputs.append(Dense(enum_size(LB.BullyingClass),
                      activation="softmax",
                      name="bullying_class")(prev_layer))
  # For each person
  for p_idx in range(num_people):
    outputs.append(Dense(1,
                        activation="sigmoid",
                        name="exists_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(simple_proto_size(LB.Bbox),
                        activation="relu",
                        name="bbox_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.EmotionClass),
                        activation="sigmoid",
                        name="discrete_emotion_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(simple_proto_size(LB.ContinuousEmotion),
                        activation="relu",
                        name="continuous_emotion_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.Role),
                        activation="softmax",
                        name="role_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.Gender),
                        activation="softmax",
                        name="gender_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.Age),
                        activation="softmax",
                        name="age_{}".format(p_idx)
                       )(prev_layer))
  return outputs

def initialize_model(config, num_people):
  if config.HasField("custom_model"):
    ValueError("Custom Models not impl")
    if config.custom_model in CUSTOM_MODELS:
      return CUSTOM_MODELS[config.custom_model](config)
    else:
      raise ValueError("Custom model not supported:" + config.custom_model)

  current_layer = input_layer = Input(
      (config.target_size.width,
       config.target_size.height,
       colormode_to_dim(config.color_mode)))

  # For each intermediate layer
  for count, layer_conf in enumerate(config.model.layers):
    if layer_conf.HasField("convolutional"):
      current_layer = add_convolutional(layer_conf.convolutional,
                                        current_layer)
    elif layer_conf.HasField("dense"):
      current_layer = Dense(layer_conf.dense.units,
                         activation=layer_conf.dense.activation,
                         name=get_or_none(layer_conf, "name")
                        )(current_layer)
    elif layer_conf.HasField("flatten"):
      current_layer = Flatten(name=get_or_none(layer_conf, "name")
                          )(current_layer)
    elif layer_conf.HasField("dropout"):
      current_layer = Dropout(layer_conf.dropout.rate,
                           name=get_or_none(layer_conf, "name")
                          )(current_layer)
    elif layer_conf.HasField("transfer"):
      if count == 0:
        current_layer = load_transfer_layers(layer_conf, current_layer)
      else:
        raise ValueError("Transfer layer must be first.")
    else:
      raise ValueError("Layer not supported.")

  outputs = get_output_layers(num_people, current_layer)
  model = Model(input=input_layer, output=outputs)
  return model


def initialize_optimizer(config):
  opt_conf = config.model.optimizer
  if opt_conf.HasField("default_string"):
    return opt_conf.default_string
  elif opt_conf.HasField("sgd"):
    return SGD(lr=opt_conf.sgd.learning_rate,
               momentum=opt_conf.sgd.momentum,
               decay=opt_conf.sgd.decay,
               nesterov=opt_conf.sgd.nesterov)
  else:
    raise ValueError("Must supply optimizer.")

def mse_nan(y_true, y_pred):
  # This loss function excludes NaN
  # Code taken from GitHub comment:
  # https://github.com/keras-team/keras/issues/9549
  index = ~K.tf.is_nan(y_true)
  y_true = K.tf.boolean_mask(y_true, index)
  y_pred = K.tf.boolean_mask(y_pred, index)
  # Need max in case of nan
  return K.maximum(K.mean((y_true - y_pred) ** 2), K.zeros(shape=(1,)))

def train_main(args):
  # Entry point into training from __main__.py

  config = get_config(args)

  train_generator = ImageAndAnnotationGenerator(
      args.data,
      split_type="TRAIN",
      num_people=config.max_people_per_img
  ).flow_from_mongo(
    sample_size=(config.target_size.width, config.target_size.height),
    short_side_size=config.short_side_size,
    batch_size=config.batch_size
  )

  if args.model.is_file():
    log.info("Loading model to continue training from %s", args.model)
    model = load_model(str(args.model))
  else:
    log.info("Creating a new model")
    model = initialize_model(config, config.max_people_per_img)

  if config.system.HasField("gpus"):
    log.info("Configuring for %s gpus", config.system.gpus)
    model = multi_gpu_model(model, gpus=config.system.gpus)

  plot_model(model, to_file='model.png')

  optimizer = initialize_optimizer(config)

  model.compile(optimizer=optimizer,
                loss=mse_nan,
                metrics=config.model.metrics)

  log.info(model.summary())

  model.fit_generator(
      train_generator,
      epochs=config.epochs,
      steps_per_epoch=config.steps_per_epoch,
#      validation_data=val_generator,
      # validation_steps=config.validation_steps,
 #     workers=get_worker_count(config),
      callbacks = [
  #      EarlyStopping(),
        TerminateOnNaN(),
        ModelCheckpoint(str(args.model), save_best_only=True),
  #      TensorBoard(update_freq="batch")
        ]
      )

  return 0 # Exit Code
