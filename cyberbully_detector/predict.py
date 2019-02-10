import logging as log
from .data_util import setup_test_data_generator
from .data_util import get_worker_count
from .data_util import get_config
from .data_util import class_name_map
from keras.models import load_model
import numpy as np

def predict_main(args):

  config = get_config(args)

  log.info("Checking %s is readable", args.model)
  assert args.model.is_file()

  log.info("Loading")
  model = load_model(str(args.model))

  generator = setup_test_data_generator(args.data, config)

  num_eval_examples = len(generator.filenames)

  # Need workers=1 in order to keep order
  predictions = model.predict_generator(
      generator,
      steps=num_eval_examples,
      )

  # warning, we don't know what the class labels are when test data is uncategorized
  for prediction, filename in zip(predictions, generator.filenames):
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    print(filename, class_idx, confidence)


  return 0 # Exit Code
