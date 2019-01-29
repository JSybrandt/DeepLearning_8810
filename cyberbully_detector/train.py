import logging as log
from .proto_util import PROTO_PARSERS
from .proto_util import get_or_none
from .bully_pb2 import TrainingConfig
from .bully_pb2 import GRAYSCALE, RGBA, RGB, RBG
from keras.preprocessing.image import ImageDataGenerator

def colormode_to_str(color_mode):
  modes = {}
  modes[GRAYSCALE] = "grayscale"
  modes[RGB] = "rgb"
  modes[RBG] = "rbg"
  modes[RGBA] = "rgba"
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

def train_main(args):
  # Entry point into training from __main__.py

  # Default Config
  config = TrainingConfig()
  if args.config is not None:
    log.info("Parsing training config")
    PROTO_PARSERS[args.config.suffix](args.config, config)

  generator = setup_data_generator(args.data_dir, config)\

  return 0 # Exit Code
