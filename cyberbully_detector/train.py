import logging as log
from .data_util import colormode_to_str
from .data_util import colormode_to_dim
from .data_util import setup_data_generator
from .data_util import get_config
from .bully_pb2 import Config
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten


def initialize_model(config, num_classes):
  num_input_param = config.target_size.width * config.target_size.height

  img_in = Input(shape=(config.target_size.width,
                        config.target_size.height,
                        colormode_to_dim(config.color_mode)))
  flat = Flatten()(img_in)
  hidden = Dense(int((num_input_param+num_classes)/2))(flat)
  cat_out = Dense(num_classes)(hidden)
  return Model(img_in, cat_out)


def train_main(args):
  # Entry point into training from __main__.py

  config = get_config(args)

  train_generator = setup_data_generator(args.train_data_dir, config)
  val_generator = setup_data_generator(args.val_data_dir, config)

  assert train_generator.class_indices == val_generator.class_indices
  num_classes = len(train_generator.class_indices)

  if args.model.is_file():
    log.info("Loading model to continue training from %s", args.model)
    model = load_model(args.model)
  else:
    log.info("Creating a new model")
    model = initialize_model(config, num_classes)

  model.compile(optimizer="sgd", loss="categorical_crossentropy")

  model.fit_generator(
      train_generator,
      epochs=config.epochs,
      steps_per_epoch=config.steps_per_epoch,
      validation_data=val_generator,
      validation_steps=config.validation_steps
      )

  log.info("Saving model architecture and weights to %s", args.model)
  model.save(str(args.model))
  return 0 # Exit Code
