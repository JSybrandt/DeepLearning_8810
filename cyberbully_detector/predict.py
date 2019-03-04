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

  log.info("Checking %s is readable", args.model)
  meta_file = list(args.model.glob("*.meta"))[0]
  assert meta_file.is_file()

  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(str(meta_file))
    new_saver.restore(sess, tf.train.latest_checkpoint(str(args.model)))

    graph = tf.get_default_graph()

    input_placeholder = graph.get_tensor_by_name("input:0")
    predicted_class_dist = graph.get_tensor_by_name("training_vars/output/Softmax:0")

    img_data = [proc_img_path(args.file_path,
                             get_or_none(config, "short_side_size"),
                             (config.target_size.width, config.target_size.height))]
    data = {input_placeholder: img_data}
    prediction = sess.run(predicted_class_dist, feed_dict=data)
    log.info(prediction)

    print(bully_enum_to_str(np.argmax(prediction)+1))

  # log.info("Loading")
  # model = load_model(str(args.model), custom_objects={"mse_nan": mse_nan})

  # path_to_fragment = {}
  # def add_frag_callback(path, img):
    # path_to_fragment[path] = img

  # generator = FileSystemImageGenerator(
      # args.data,
      # sample_size=(config.target_size.width, config.target_size.height),
      # short_side_size=get_or_none(config, "short_side_size"),
      # batch_size=config.batch_size,
      # img_callbacks=[add_frag_callback])


  # # Need workers=1 in order to keep order
  # predictions = model.predict_generator(
      # generator,
      # #  workers=get_worker_count(config)
      # )

  # for prediction_vec, file_path in zip(predictions, generator.get_files()):
    # annotation = vector_to_annotation(prediction_vec)
    # print(file_path, annotation)
    # if args.out_dir is not None:
      # if args.write_annotations:
        # annotation_path = args.out_dir.joinpath(file_path.stem + ".annotation")
        # with open(annotation_path, "rb") as file:
          # file.write(annotation_path.SerializeToString())
      # if args.write_images:
        # img_path  = args.out_dir.joinpath(file_path.stem + ".jpg")
        # img = path_to_fragment[file_path]
        # unscale_people(img, annotation)
        # img = draw_boxes_on_image(img, annotation)
        # img.save(str(img_path))

  return 0 # Exit Code
