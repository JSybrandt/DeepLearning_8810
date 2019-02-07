import logging as log
from .data_util import colormode_to_str
from .data_util import colormode_to_dim
from .data_util import setup_training_data_generator
from .data_util import setup_eval_data_generator
from .data_util import get_worker_count
from .data_util import get_config
from .bully_pb2 import Config
from .proto_util import get_or_none
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import multi_gpu_model

def add_convolutional(conv_config, last_layer):
  name=get_or_none(conv_config, "name")
  last_layer = Conv2D(
      conv_config.filters,
      (conv_config.kernel_size.width, conv_config.kernel_size.height),
      name=name
  )(last_layer)

  if name is not None:
    name += "_pool"

  if conv_config.pool_type == "max":
    last_layer = MaxPooling2D(
        (conv_config.pool_size.width, conv_config.pool_size.height),
        name=name
    )(last_layer)
  else:
    raise ValueError("Unsupported pool type:" + conv_config.pool_type)
  return last_layer

def initialize_model(config, num_classes):
  # Variable sized input
  input_layer = Input(
      (config.target_size.width,
       config.target_size.height,
       colormode_to_dim(config.color_mode)))

  last_layer = input_layer

  # For each intermediate layer
  for layer in config.model.layers:
    if layer.HasField("convolutional"):
      last_layer = add_convolutional(layer.convolutional, last_layer)
    elif layer.HasField("dense"):
      last_layer = Dense(layer.dense.units,
                         activation=layer.dense.activation,
                         name=get_or_none(layer.dense, "name")
                        )(last_layer)
    elif layer.HasField("flatten"):
      last_layer = Flatten(name=get_or_none(layer.flatten, "name")
                          )(last_layer)
    elif layer.HasField("dropout"):
      last_layer = Dropout(layer.dropout.rate,
                           name=get_or_none(layer.dropout, "name")
                          )(last_layer)

  model = Model(input_layer, last_layer)
  return model


def train_main(args):
  # Entry point into training from __main__.py

  config = get_config(args)

  log.info(config)

  train_generator = setup_training_data_generator(args.train_data_dir, config)
  val_generator = setup_eval_data_generator(args.val_data_dir, config)

  assert train_generator.class_indices == val_generator.class_indices
  num_classes = len(train_generator.class_indices)

  if args.model.is_file():
    log.info("Loading model to continue training from %s", args.model)
    model = load_model(str(args.model))
  else:
    log.info("Creating a new model")
    model = initialize_model(config, num_classes)

  if config.system.HasField("gpus"):
    log.info("Configuring for %s gpus", config.system.gpus)
    model = multi_gpu_model(model, gpus=config.system.gpus)

  model.compile(optimizer=config.model.optimizer,
                loss=config.model.loss)

  log.info(model.summary())


  model.fit_generator(
      train_generator,
      epochs=config.epochs,
      steps_per_epoch=config.steps_per_epoch,
      validation_data=val_generator,
      validation_steps=config.validation_steps,
      workers=get_worker_count(config),
      )

  log.info("Saving model architecture and weights to %s", args.model)
  model.save(str(args.model))
  return 0 # Exit Code
