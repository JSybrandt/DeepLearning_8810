from .bully_pb2 import TrainingConfig
from .train import train_main
from .proto_util import PROTO_PARSERS

from argparse import ArgumentParser
from pathlib import Path
import logging as log
import sys
from collections import namedtuple

ArgumentCheck = namedtuple("ArgumentCheck", ["command",
                                             "param_name",
                                             "assert_fn",
                                             "err_msg"])

def is_safe_to_write_file(path, overwritable=False):
  if path.is_file():
    return overwritable
  return not path.exists() and path.parent.is_dir()

def is_safe_to_read(path):
  return path.is_file()

################################################################################

def parse_args():
  checks = []

  ## Configure parser options ##################################################
  root_parser = ArgumentParser(description="JSybrandt's submission for 8810.")

  # Configure command sub-parsers
  subparsers = root_parser.add_subparsers()
  train_parser = subparsers.add_parser(
      "train",
      description="Train and save a model")
  eval_parser = subparsers.add_parser(
      "eval",
      description="Evaluate an existing model")

  # command default lets me know what command was run
  root_parser.set_defaults(command="root")
  train_parser.set_defaults(command="train")
  eval_parser.set_defaults(command="eval")

  ## Root Arguments - Common for all commands ##################################
  root_parser.add_argument("-v", "--verbose", action="store_true")
  root_parser.add_argument("--debug", action="store_true")

  root_parser.add_argument("-l", "--log_path", type=Path)
  checks.append(ArgumentCheck(
    command="all",
    param_name="log_path",
    assert_fn=lambda a: (a.log_path is None \
                         or is_safe_to_write_file(a.log_path, True)),
    err_msg="Must file path must be writable."
  ))

  ## Training Arguments ########################################################

  train_parser.add_argument("--config", type=Path)
  checks.append(ArgumentCheck(
    command="train",
    param_name="config",
    assert_fn=lambda a: (\
        a.config is None or (\
          a.config.is_file() and \
          a.config.suffix in PROTO_PARSERS)),
    err_msg="Config must exist and be one of these extensions: " \
            + ", ".join(PROTO_PARSERS)
  ))

  train_parser.add_argument("data_dir",
                            type=Path,
                            nargs="?",
                            default=Path("./data/train"))
  checks.append(ArgumentCheck(
    command="train",
    param_name="data_dir",
    assert_fn=lambda a: a.data_dir.is_dir(),
    err_msg="Must supply a directory."
  ))

  train_parser.add_argument("model_path",
                            type=Path,
                            nargs="?",
                            default=Path("model.h5"))
  checks.append(ArgumentCheck(
    command="train",
    param_name="model_path",
    assert_fn=lambda a: (\
          a.model_path.suffix == ".h5" and \
          is_safe_to_write_file(a.model_path)),
    err_msg="Must supply writable .h5 file."
  ))

  ## Evaluation / Testing Arguments ############################################
  #TODO(JSybran)

  args = root_parser.parse_args()

  has_error = False
  for check in checks:
    if check.command in [args.command, "all"] \
        and not check.assert_fn(args):
      print(">> Parameter Error:",
            check.param_name,
            "=",
            getattr(args, check.param_name),
            file=sys.stderr)
      print("--", check.err_msg, file=sys.stderr)
      has_error = True
  if has_error:
    exit(1)

  return args

def config_logger(args):
  logger = log.getLogger()
  formatter = log.Formatter(
      "%(asctime)s %(levelname)s: %(message)s",
      "%H:%M")

  # SET LEVEL
  if args.verbose:
    logger.setLevel(log.INFO)
  if args.debug:
    logger.setLevel(log.DEBUG)

  # Log to StdErr
  handler = log.StreamHandler()
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  # Log to file
  if args.log_path is not None:
    handler = log.FileHandler(args.log_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__ == "__main__":
  args = parse_args()
  config_logger(args)
  logger = log.getLogger()
  logger.info("START %s", str(args))

  if args.command == "train":
    exit(train_main(args))
  if args.command == "eval":
    logger.error("Eval not implemented.")
    exit(1)
  logger.error("Invalid command")
  exit(1)
