import logging as log
from .proto_util import PROTO_PARSERS
from .proto_util import get_or_none
from .bully_pb2 import TrainingConfig
from .bully_pb2 import GRAYSCALE, RGBA, RGB, RBG
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten

def colormode_to_str(color_mode):
  modes = {}
  modes[GRAYSCALE] = "grayscale"
  modes[RGB] = "rgb"
  modes[RBG] = "rbg"
  modes[RGBA] = "rgba"
  assert color_mode in modes
  return modes[color_mode]

def colormode_to_dim(color_mode):
  modes = {}
  modes[GRAYSCALE] = 1
  modes[RGB] = 3
  modes[RBG] = 3
  modes[RGBA] = 4
  assert color_mode in modes
  return modes[color_mode]

def setup_data_generator(data_path, config):
  return ImageDataGenerator(
      horizontal_flip=config.datagen.horizontal_flip,
      vertical_flip=config.datagen.vertical_flip,
      shear_range=config.datagen.shear_range,
      zoom_range=config.datagen.zoom_range,
      width_shift_range=config.datagen.width_shift_range,
      height_shift_range=config.datagen.height_shift_range,
      rotation_range=config.datagen.rotation_range,
      validation_split=config.datagen.validation_split,
  ).flow_from_directory(
      data_path,
      target_size=(config.target_size.width,
                   config.target_size.height),
      batch_size=config.batch_size,
      class_mode="categorical",
      color_mode=colormode_to_str(config.color_mode),
      shuffle=config.shuffle_input,
      save_to_dir=get_or_none(config, "vis_result_dir"),
      seed=get_or_none(config, "seed"),
  )


def get_model(config, num_classes):
  num_input_param = config.target_size.width * config.target_size.height

  img_in = Input(shape=(config.target_size.width,
                        config.target_size.height,
                        colormode_to_dim(config.color_mode)))
  flat = Flatten()(img_in)
  hidden_1 = Dense(num_input_param)(flat)
  hidden_2 = Dense(int((num_input_param+num_classes)/2))(hidden_1)
  cat_out = Dense(num_classes)(hidden_2)
  return Model(img_in, cat_out)


def train_main(args):
  # Entry point into training from __main__.py

  # Default Config
  config = TrainingConfig()
  if args.config is not None:
    log.info("Parsing training config")
    PROTO_PARSERS[args.config.suffix](args.config, config)

  train_generator = setup_data_generator(args.train_data_dir, config)
  val_generator = setup_data_generator(args.val_data_dir, config)

  assert train_generator.class_indices == val_generator.class_indices
  num_classes = len(train_generator.class_indices)

  model = get_model(config, num_classes)

  model.compile(optimizer="adagrad", loss="categorical_crossentropy")

  model.fit_generator(
      train_generator,
      epochs=config.epochs,
      steps_per_epoch=config.steps_per_epoch,
      validation_data=val_generator,
      validation_steps=config.validation_steps
      )

  return 0 # Exit Code
