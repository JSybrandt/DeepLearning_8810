import logging as log
from .data_util import get_worker_count
from .data_util import get_config
import numpy as np

def evaluate_main(args):
  # Entry point into eval from __main__.py

  config = get_config(args)

  log.info("Checking %s is readable", args.model)
  assert args.model.is_file()

  log.info("Loading")
  # model = load_model(str(args.model))

  # test_generator = setup_test_data_generator(args.data, config)

  # num_eval_examples = len(test_generator.filenames)

  # # Need workers=1 in order to keep order
  # predictions = model.predict_generator(
      # test_generator,
      # steps=num_eval_examples,
      # )

  # true_labels = test_generator.classes

  # predicted_labels = [np.argmax(p) for p in predictions]

  # label_names = [c for c in test_generator.class_indices]
  # label_names.sort()
  # print(label_names)

  # assert len(true_labels) == len(predicted_labels)

  # print(classification_report(true_labels,
                              # predicted_labels,
                              # target_names=label_names))


  return 0 # Exit Code
