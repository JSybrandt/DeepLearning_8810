import logging as log
from .proto_util import PROTO_PARSERS
from .bully_pb2 import TrainingConfig

def index_labels(config):
  log.info("Indexing labels from training data.")
  config.label_map.clear()
  for example in config.examples:
    if example.label not in config.label_map:
      config.label_map[example.label] = len(config.label_map)


def train_main(args):
  # Entry point into training from __main__.py
  log.info("Parsing training config")
  config = TrainingConfig()
  PROTO_PARSERS[args.config.suffix](args.config, config)
  return 0 # Exit Code
