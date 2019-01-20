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



def parse_args():
  checks = []

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
  checks.append(ArgumentCheck(
    command="all",
    param_name="command",
    assert_fn=lambda a: a.command != "root",
    err_msg="Must supply command"
  ))

  ### ROOT ARGS ###
  root_parser.add_argument("-v", "--verbose", action="store_true")
  root_parser.add_argument("--debug", action="store_true")
  # No check needed

  root_parser.add_argument("-l", "--log_path", type=Path)
  checks.append(ArgumentCheck(
    command="all",
    param_name="log_path",
    assert_fn=lambda a: (a.log_path is None \
                         or is_safe_to_write_file(a.log_path, True)),
    err_msg="Must file path must be writable."
  ))

  ### TRAIN ARGS ###
  train_parser.add_argument("data", type=Path)
  checks.append(ArgumentCheck(
    command="train",
    param_name="data",
    assert_fn=lambda a: a.data.is_dir(),
    err_msg="Must supply existing directory."
  ))

  train_parser.add_argument("model_path", type=Path)
  checks.append(ArgumentCheck(
    command="train",
    param_name="data",
    assert_fn=lambda a: (a.model_path.suffix == ".h5" \
                         and is_safe_to_write_file(a.model_path)),
    err_msg="Must supply writable .h5 file."
  ))

  ### EVAL ARGS ###
  #TODO(JSybran)

  args = root_parser.parse_args()
  print(args)

  for check in checks:
    if check.command in [args.command, "all"] \
        and not check.assert_fn(args):
      print("Error in parameter:",
            check.param_name,
            "=",
            getattr(args, check.param_name),
            file=sys.stderr)
      print(">>>", check.err_msg, file=sys.stderr)
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
