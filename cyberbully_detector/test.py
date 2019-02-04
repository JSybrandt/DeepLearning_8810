import logging as log
from .data_util import setup_data_generator
from keras.models import load_model
from .data_util import get_config

def test_main(args):
  # Entry point into eval from __main__.py

  config = get_config(args)

  log.info("Checking %s is readable", args.model)
  assert args.model.is_file()

  log.info("Loading")
  model = load_model(str(args.model))

  test_generator = setup_data_generator(args.test_data_dir, config)

  predictions = model.predict_generator(test_generator)

  print(predictions)

  return 0 # Exit Code
