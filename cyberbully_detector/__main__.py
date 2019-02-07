from .train import train_main
from .evaluate import evaluate_main
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

################################################################################

def parse_args():
  checks = []

  ## Configure parser options ##################################################
  root_parser = ArgumentParser(description="JSybrandt's submission for 8810.")

  ## Root Arguments - Common for all commands ##################################
  root_parser.add_argument("model",
                           type=Path,
                           help="Path of *.h5 to save/load model")
  checks.append(ArgumentCheck(
    command="all",
    param_name="model",
    assert_fn=lambda a: (is_safe_to_write_file(a.model, True)
                         and a.model.suffix == ".h5"),
    err_msg="File path must be writable *.h5."
  ))

  root_parser.add_argument("-v", "--verbose", action="store_true")

  root_parser.add_argument("--debug", action="store_true")

  root_parser.add_argument("-d",
                           "--data_root",
                           type=Path,
                           default="./data",
                           help="Base directory for training data.")
  checks.append(ArgumentCheck(
    command="all",
    param_name="data_root",
    assert_fn=lambda a: a.data_root.is_dir(),
    err_msg="Directory not found."
  ))

  root_parser.add_argument("-l", "--log_path", type=Path)
  checks.append(ArgumentCheck(
    command="all",
    param_name="log_path",
    assert_fn=lambda a: (a.log_path is None \
                         or is_safe_to_write_file(a.log_path, True)),
    err_msg="File path must be writable."
  ))

  root_parser.add_argument("--config", type=Path)
  checks.append(ArgumentCheck(
    command="all",
    param_name="config",
    assert_fn=lambda a: (\
        a.config is None or (\
          a.config.is_file() and \
          a.config.suffix in PROTO_PARSERS)),
    err_msg="Config must exist and be one of these extensions: " \
            + ", ".join(PROTO_PARSERS)
  ))

  # Configure command sub-parsers
  subparsers = root_parser.add_subparsers()
  train_parser = subparsers.add_parser(
      "train",
      description="Train and save a model")
  test_parser = subparsers.add_parser(
      "evaluate",
      description="Evaluate an existing model")

  # command default lets me know what command was run
  root_parser.set_defaults(command="root")
  train_parser.set_defaults(command="train")
  test_parser.set_defaults(command="evaluate")

  ## Training Arguments ########################################################

  train_parser.add_argument("train_data_dir",
                            type=str,
                            nargs="?",
                            default="train",
                            help="Path relative to data_root")
  checks.append(ArgumentCheck(
    command="train",
    param_name="train_data_dir",
    assert_fn=lambda a: a.data_root.joinpath(a.train_data_dir).is_dir(),
    err_msg="Cannot find directory relative to data_root"
  ))

  train_parser.add_argument("val_data_dir",
                            type=str,
                            nargs="?",
                            default="validation",
                            help="Path relative to data_root")
  checks.append(ArgumentCheck(
    command="train",
    param_name="val_data_dir",
    assert_fn=lambda a: a.data_root.joinpath(a.val_data_dir).is_dir(),
    err_msg="Cannot find directory relative to data_root"
  ))

  ## Evaluation / Testing Arguments ############################################
  test_parser.add_argument("test_data_dir",
                            type=str,
                            nargs="?",
                            default="evaluate",
                            help="Path relative to data_root")
  checks.append(ArgumentCheck(
    command="evaluate",
    param_name="test_data_dir",
    assert_fn=lambda a: a.data_root.joinpath(a.test_data_dir).is_dir(),
    err_msg="Cannot find directory relative to data_root"
  ))

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

  if hasattr(args, "train_data_dir"):
    args.train_data_dir=args.data_root.joinpath(args.train_data_dir)
  if hasattr(args, "val_data_dir"):
    args.val_data_dir=args.data_root.joinpath(args.val_data_dir)
  if hasattr(args, "test_data_dir"):
    args.test_data_dir=args.data_root.joinpath(args.test_data_dir)

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
  if args.command == "evaluate":
    exit(evaluate_main(args))
  logger.error("Invalid command")
  exit(1)
