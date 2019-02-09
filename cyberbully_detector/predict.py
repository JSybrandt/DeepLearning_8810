import logging as log
from .data_util import setup_eval_data_generator
from .data_util import get_worker_count
from .data_util import get_config
from keras.models import load_model

def predict_main(args):

  config = get_config(args)

  log.info("Checking %s is readable", args.model)
  assert args.model.is_file()

  log.info("Loading")
  model = load_model(str(args.model))

  generator = setup_eval_data_generator(args.data, config)

  num_eval_examples = len(generator.filenames)

  predictions = model.predict_generator(
      generator,
      workers=get_worker_count(config),
      steps=num_eval_examples,
      )

  log.info("Predictions:")
  log.info(predictions)
  log.info(predictions.shape)

  return 0 # Exit Code
