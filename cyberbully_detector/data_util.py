from .bully_pb2 import GRAYSCALE, RGBA, RGB, RBG
from .bully_pb2 import Config
from .proto_util import PROTO_PARSERS
from .proto_util import get_or_none
from keras.preprocessing.image import ImageDataGenerator
import logging as log
import multiprocessing

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

def setup_training_data_generator(data_path, config):
  return ImageDataGenerator(
      horizontal_flip=config.datagen.horizontal_flip,
      vertical_flip=config.datagen.vertical_flip,
      shear_range=config.datagen.shear_range,
      zoom_range=config.datagen.zoom_range,
      width_shift_range=config.datagen.width_shift_range,
      height_shift_range=config.datagen.height_shift_range,
      rotation_range=config.datagen.rotation_range,
      validation_split=config.validation_split,
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
      follow_links=True
  )

def setup_eval_data_generator(data_path, config):
  return ImageDataGenerator(
  ).flow_from_directory(
      data_path,
      target_size=(config.target_size.width,
                   config.target_size.height),
      batch_size=1,
      class_mode=config.class_mode,
      color_mode=colormode_to_str(config.color_mode),
      shuffle=False,
      seed=get_or_none(config, "seed"),
      follow_links=True
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

def get_worker_count(config):
  total_cpus = multiprocessing.cpu_count()
  desired_cpu = config.system.workers
  if desired_cpu < 0:
    desired_cpu += total_cpus
  if desired_cpu < 0:
    raise ValueError("Config is asking for fewer workers than available")
  if desired_cpu > total_cpus:
    raise ValueError("Config is asking for more workers than available")
  log.info("Using %s/%s workers", desired_cpu, total_cpus)
  return desired_cpu
