from tensorflow import keras
from .bully_pb2 import GRAYSCALE, RGBA, RGB, RBG
from .bully_pb2 import Config
from .proto_util import PROTO_PARSERS
from .proto_util import get_or_none
from keras.preprocessing.image import ImageDataGenerator
import logging as log

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
      class_mode=config.class_mode,
      color_mode=colormode_to_str(config.color_mode),
      shuffle=config.shuffle_input,
      save_to_dir=get_or_none(config, "vis_result_dir"),
      seed=get_or_none(config, "seed"),
  )

def get_config(args):
  # Default Config
  config = Config()
  if args.config is not None:
    log.info("Parsing config")
    PROTO_PARSERS[args.config.suffix](args.config, config)
  else:
    log.info("Using default config")
  return config
