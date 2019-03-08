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


def predict_with_model(model_dir, img_data):
  log.info("Loading %s", model_dir)
  meta_file = list(model_dir.glob("*.meta"))[0]
  assert meta_file.is_file()

  tf.reset_default_graph()
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(str(meta_file))
    new_saver.restore(sess, tf.train.latest_checkpoint(str(model_dir)))

    graph = tf.get_default_graph()

    input_placeholder = graph.get_tensor_by_name("input:0")
    predicted_class_dist = graph.get_tensor_by_name("trainable_vars/output/Softmax:0")
    is_training = graph.get_tensor_by_name("PlaceholderWithDefault/input:0")


    data = {input_placeholder: img_data, is_training:0}
    predictions = sess.run(predicted_class_dist, feed_dict=data)
    return predictions

def predict_main(args):

  config = get_config(args)
  models = [args.model]
  if args.sup_model is not None:
    models += args.sup_model
  if args.last_ditch_model is not None:
    models += args.last_ditch_model

  img_data = [
      proc_img_path(p,
                    get_or_none(config, "short_side_size"),
                    (config.target_size.width, config.target_size.height),
                    flips=False,
                    force_horizontal_flip=False,
                    force_vertical_flip=False)
      for p in args.file_path
  ]

  num_classes = len(LB.BullyingClass.keys())
  predictions = np.zeros(shape=(len(img_data), num_classes))

  for model_dir in models:
    predictions += predict_with_model(model_dir, img_data)

  labels = np.argmax(predictions, axis=1) + 1

  if len(args.file_path) > 1:
    for path, lab in zip(args.file_path, labels):
      print(path, bully_enum_to_str(lab))
  else:
    print(bully_enum_to_str(labels[0]))

  return 0 # Exit Code
