from .train import train_main
from .evaluate import evaluate_main
from .predict import predict_main
from .proto_util import PROTO_PARSERS
from argparse import ArgumentParser
from pathlib import Path
import logging as log
import sys
from collections import namedtuple
import os

# disable tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

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
  root_parser.add_argument("config", type=Path)
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

  root_parser.add_argument("model",
                           type=Path,
                           help="Dir to save/load model")
  checks.append(ArgumentCheck(
    command="all",
    param_name="model",
    assert_fn=lambda a: ((not a.model.exists() and a.model.parent.is_dir())\
                         or a.model.is_dir()),
    err_msg="Must be able to create a directory here."
  ))

  root_parser.add_argument("-v", "--verbose", action="store_true")

  root_parser.add_argument("--debug", action="store_true")

  root_parser.add_argument("-l", "--log_path", type=Path)
  checks.append(ArgumentCheck(
    command="all",
    param_name="log_path",
    assert_fn=lambda a: (a.log_path is None \
                         or is_safe_to_write_file(a.log_path, True)),
    err_msg="File path must be writable."
  ))

  root_parser.add_argument("--data",
                            type=Path,
                            help="Location of data. Subdir per data source",
                            default="./data")
  checks.append(ArgumentCheck(
    command="root",
    param_name="data",
    assert_fn=lambda a: a.data_dir.is_dir(),
    err_msg="Cannot find directory"
  ))

  # Configure command sub-parsers
  subparsers = root_parser.add_subparsers()
  train_parser = subparsers.add_parser(
      "train",
      description="Train and save a model.")
  eval_parser = subparsers.add_parser(
      "evaluate",
      description="Load model, compare predictions against know classes.")
  predict_parser = subparsers.add_parser(
      "predict",
      description="Load model, output predictions for unknown data.")

  # command default lets me know what command was run
  root_parser.set_defaults(command="root")
  train_parser.set_defaults(command="train")
  eval_parser.set_defaults(command="evaluate")
  predict_parser.set_defaults(command="predict")

  ## Training Arguments ########################################################

  train_parser.add_argument(
      "--dataset", type=str,
      help="If set, only use specific dataset images for training.",
      nargs="*")

  train_parser.add_argument(
      "--resume", action="store_true",
      help="If set, load and continue training model")

  train_parser.add_argument(
      "--fold", type=int,
      help="Which fold to hold out. If set, ignores dataclass labels.")
  checks.append(ArgumentCheck(
    command="train",
    param_name="fold",
    assert_fn=lambda a: a.fold is None or a.fold >= 0,
    err_msg="If supplied, fold must be non-negative."
  ))
  checks.append(ArgumentCheck(
    command="train",
    param_name="fold",
    assert_fn=lambda a: (a.fold is None) == (a.num_folds is None),
    err_msg="If fold is supplied. num_folds must also be supplied."
  ))

  train_parser.add_argument(
      "--num_folds", type=int,
      help="Which fold to hold out. If set, ignores dataclass labels.")
  checks.append(ArgumentCheck(
    command="train",
    param_name="num_folds",
    assert_fn=lambda a: a.num_folds is None or a.num_folds >= 0,
    err_msg="If supplied, num_folds must be non-negative."
  ))
  checks.append(ArgumentCheck(
    command="train",
    param_name="num_folds",
    assert_fn=lambda a: a.fold is None or a.num_folds > a.fold,
    err_msg="Max fold must be larger than fold."
  ))

  ## Evaluation / Arguments ###################################################
  ## Prediction / Arguments ###################################################

  predict_parser.add_argument("--out_dir", type=Path)
  checks.append(ArgumentCheck(
    command="predict",
    param_name="out_dir",
    assert_fn=lambda a: a.out_dir is None or a.out_dir.is_dir(),
    err_msg="Cannot find directory"
    ))

  predict_parser.add_argument("--write_annotations", action="store_true")
  checks.append(ArgumentCheck(
    command="predict",
    param_name="write_annotations",
    assert_fn=lambda a: ((not a.write_annotations) \
                         or a.write_annotations == (a.out_dir is not None)),
    err_msg="If --write_annotations is set, --out_dir must also be set"
  ))

  predict_parser.add_argument("--write_images", action="store_true")
  checks.append(ArgumentCheck(
    command="predict",
    param_name="write_images",
    assert_fn=lambda a: ((not a.write_images) \
                         or a.write_images == (a.out_dir is not None)),
    err_msg="If --write_images is set, --out_dir must also be set"
  ))

  predict_parser.add_argument("file_path", type=Path, nargs="+")
  checks.append(ArgumentCheck(
    command="predict",
    param_name="file_path",
    assert_fn=lambda a: (sum([p.is_file() for p in a.file_path]) \
                         == len(a.file_path)),
    err_msg="Must supply valid path."
  ))

  predict_parser.add_argument("--sup_model", type=Path, nargs="*")
  checks.append(ArgumentCheck(
    command="predict",
    param_name="sup_model",
    assert_fn=lambda a: (a.sup_model is None or \
                         sum([p.is_dir() for p in a.sup_model]) \
                         == len(a.sup_model)),
    err_msg="Must supply valid paths for supplemental models."
  ))

  predict_parser.add_argument("--last_ditch_model", type=Path, nargs="*")
  checks.append(ArgumentCheck(
    command="predict",
    param_name="last_ditch_model",
    assert_fn=lambda a: (a.last_ditch_model is None or \
                         sum([p.is_dir() for p in a.last_ditch_model]) \
                         == len(a.last_ditch_model)),
    err_msg="Must supply valid paths for last-ditch models."
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

  return args

def config_logger(args):
  logger = log.getLogger()
  formatter = log.Formatter(
      "%(asctime)s %(levelname)s: %(message)s",
      "%H:%M")

  # SET LEVEL
  if args.verbose:
    logger.setLevel(log.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  if args.debug:
    logger.setLevel(log.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

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
  if args.command == "predict":
    exit(predict_main(args))
  logger.error("Invalid command")
  exit(1)
