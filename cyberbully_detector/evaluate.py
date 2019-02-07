import logging as log
from .data_util import setup_eval_data_generator
from .data_util import get_worker_count
from .data_util import get_config
from keras.models import load_model

def evaluate_main(args):
  # Entry point into eval from __main__.py

  config = get_config(args)

  log.info("Checking %s is readable", args.model)
  assert args.model.is_file()

  log.info("Loading")
  model = load_model(str(args.model))

  test_generator = setup_eval_data_generator(args.test_data_dir, config)

  num_eval_examples = len(test_generator.filenames)

  predictions = model.evaluate_generator(
      test_generator,
      steps=num_eval_examples,
      workers=get_worker_count(config),
      )

  print(predictions)

  return 0 # Exit Code
