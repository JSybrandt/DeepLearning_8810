import logging as log
from .generator import proc_img_path
from .data_util import get_worker_count
from .data_util import get_config
from .data_util import class_name_map
from .proto_util import vector_to_annotation
from .proto_util import vec_to_enum
from .proto_util import get_or_none
from .proto_util import unscale_people
from .visualize import draw_boxes_on_image
import numpy as np
from pathlib import Path
import tensorflow as tf
from . import labels_pb2 as LB


def bully_enum_to_str(bully_enum_val):
  _val_map = {
    LB.GOSSIPING: "gossiping",
    LB.LAUGHING: "laughing",
    LB.PULLING_HAIR: "pullinghair",
    LB.QUARRELING: "quarrel",
    LB.STABBING: "stabbing",
    LB.ISOLATION: "isolation",
    LB.PUNCHING: "punching",
    LB.SLAPPING: "slapping",
    LB.STRANGLING: "strangle",
    LB.NO_BULLYING: "nonbullying",
  }
  assert bully_enum_val in _val_map
  return _val_map[bully_enum_val]

def predict_main(args):

  config = get_config(args)
  img_data = [
      proc_img_path(args.file_path,
                    get_or_none(config, "short_side_size"),
                    (config.target_size.width, config.target_size.height),
                    flips=False,
                    force_horizontal_flip=False,
                    force_vertical_flip=False),
      proc_img_path(args.file_path,
                    get_or_none(config, "short_side_size"),
                    (config.target_size.width, config.target_size.height),
                    flips=False,
                    force_horizontal_flip=True,
                    force_vertical_flip=False),
      proc_img_path(args.file_path,
                    get_or_none(config, "short_side_size"),
                    (config.target_size.width, config.target_size.height),
                    flips=False,
                    force_horizontal_flip=False,
                    force_vertical_flip=True),
      proc_img_path(args.file_path,
                    get_or_none(config, "short_side_size"),
                    (config.target_size.width, config.target_size.height),
                    flips=False,
                    force_horizontal_flip=True,
                    force_vertical_flip=True),
  ]

  predictions = []

  models = [args.model] + args.sup_model
  for model_dir in models:

    meta_file = list(model_dir.glob("*.meta"))[0]
    assert meta_file.is_file()

    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph(str(meta_file))
      new_saver.restore(sess, tf.train.latest_checkpoint(str(model_dir)))

      graph = tf.get_default_graph()

      input_placeholder = graph.get_tensor_by_name("input:0")
      predicted_class_dist = graph.get_tensor_by_name("trainable_vars/output/Softmax:0")
      is_training = graph.get_tensor_by_name("PlaceholderWithDefault/input:0")


      data = {input_placeholder: img_data, is_training:0}
      batch = sess.run(predicted_class_dist, feed_dict=data)
      # Normalize prediction by batch
      prediction = np.sum(batch, axis=0) / batch.shape[0]
      # log.info(prediction)
      predictions.append(prediction)
      log.info("Model %s beleives its %s",
               model_dir.name,
               bully_enum_to_str(np.argmax(prediction)+1))

  log.info("Global")
  prediction = np.sum(predictions, axis=0)
  log.info(prediction)
  print(bully_enum_to_str(np.argmax(prediction)+1))

  return 0 # Exit Code
