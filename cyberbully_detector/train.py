import logging as log
from .data_util import colormode_to_str
from .data_util import colormode_to_dim
from .data_util import get_worker_count
from .data_util import get_config
from .bully_pb2 import Config
from .proto_util import get_or_none
from .proto_util import annotation_size
from .generator import ImageAndAnnotationGenerator
from . import labels_pb2 as LB
from .proto_util import enum_size
from .proto_util import simple_proto_size

import tensorflow as tf

from tensorflow import placeholder as Input
from tensorflow import concat as Concatenate
from tensorflow.layers import Dense
from tensorflow.layers import Conv2D
from tensorflow.layers import MaxPooling2D
from tensorflow.layers import Dropout
from tensorflow.layers import Flatten
from tensorflow.train import MomentumOptimizer
from tensorflow.train import AdamOptimizer
from tensorflow.metrics import accuracy
from tensorflow.metrics import mean_squared_error as mse_watch

from tensorflow.train import Saver
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tempfile import TemporaryDirectory


def str_to_activation(activation_str):
  _activations = {
      "sigmoid": tf.nn.sigmoid,
      "relu": tf.nn.relu,
      "softmax": tf.nn.softmax,
      "tanh": tf.nn.tanh,
  }
  assert activation_str in _activations
  return _activations[activation_str]

def initialize_model(config, num_people):
  # START WITH INPUT
  current_layer = input_placeholder = Input(
      shape=(None, # variable batch size
             config.target_size.width,
             config.target_size.height,
             colormode_to_dim(config.color_mode)),
      dtype=tf.float32)

  # For each intermediate layer
  for count, layer_conf in enumerate(config.model.layers):
    name = get_or_none(layer_conf, "name")

    if layer_conf.HasField("convolutional"):
      current_layer = Conv2D(
          layer_conf.convolutional.filters,
          (layer_conf.convolutional.kernel_size.width,
           layer_conf.convolutional.kernel_size.height),
          name=name
      )(current_layer)

    elif layer_conf.HasField("pool"):
      if layer_conf.pool.type == "max":
        current_layer = MaxPooling2D(
            (layer_conf.pool.size.width, layer_conf.pool.size.height),
            strides=(layer_conf.pool.size.width, layer_conf.pool.size.height),
            name=name
        )(current_layer)
      else:
        raise ValueError("Unsupported pool type:" + conv_config.pool_type)

    elif layer_conf.HasField("dense"):
      current_layer = Dense(
          layer_conf.dense.units,
          activation=str_to_activation(layer_conf.dense.activation),
          name=name
          )(current_layer)

    elif layer_conf.HasField("flatten"):
      current_layer = Flatten(name=name)(current_layer)

    elif layer_conf.HasField("dropout"):
      current_layer = Dropout(layer_conf.dropout.rate,
                              name=name)(current_layer)

    elif layer_conf.HasField("transfer"):
      if count == 0:
        current_layer = load_transfer_layers(layer_conf, current_layer)
      else:
        raise ValueError("Transfer layer must be first.")
    else:
      raise ValueError("Layer not supported.")

  return input_placeholder, current_layer

def train_main(args):
  # Entry point into training from __main__.py

  config = get_config(args)

  if config.system.HasField("gpus"):
    raise ValueError("Multi-gpu support not yet impl")

  train_generator = ImageAndAnnotationGenerator(
      data_path=args.data,
      data_class="TRAIN",
      num_people=config.max_people_per_img,
      sample_size=(config.target_size.width, config.target_size.height),
      short_side_size=get_or_none(config, "short_side_size"),
      batch_size=config.batch_size,
      datasets=args.dataset
  )
  val_generator = ImageAndAnnotationGenerator(
      data_path=args.data,
      data_class="VALIDATION",
      num_people=config.max_people_per_img,
      sample_size=(config.target_size.width, config.target_size.height),
      short_side_size=get_or_none(config, "short_side_size"),
      batch_size=config.batch_size,
      datasets=args.dataset
  )

  output_size = annotation_size(config.max_people_per_img);

  total_classes = enum_size(LB.BullyingClass)

  log.info("Creating a new model")
  input_placeholder, prediction = initialize_model(config,
                                                   config.max_people_per_img)

  label_placeholder = Input(shape=(None, output_size),
                            dtype=tf.float32)

  predicted_bullying_type = tf.argmax(prediction)
  label_bullying_type = tf.argmax(label_placeholder)

  # nan_mask = ~tf.is_nan(label_placeholder)
  # masked_label = tf.boolean_mask(label_placeholder, nan_mask)
  # masked_output = tf.boolean_mask(prediction, nan_mask)
  # loss = tf.losses.softmax_cross_entropy(masked_label, masked_output);
  loss = tf.losses.softmax_cross_entropy(label_placeholder, prediction)

  acc_watch, acc_upop = accuracy(label_bullying_type,
                                 predicted_bullying_type,
                                 name="acc_watch")
  acc_init = tf.variables_initializer(
      var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                 scope="acc_watch"))

  train_acc = tf.summary.scalar("Training K-class Acc", acc_watch)
  val_acc = tf.summary.scalar("Validation K-class Acc", acc_watch)

  optimizer = AdamOptimizer(learning_rate=0.001).minimize(loss)

  # BELOW IS THE ACTUAL RUNNING

  tensorboard_dir = TemporaryDirectory()
  try:
    # ACTUALLY OPTIMIZE!
    with tf.Session() as sess:
      print("Follow along with tensorboard:", tensorboard_dir.name)
      summary_writer = tf.summary.FileWriter(tensorboard_dir.name, sess.graph)

      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      # Go super fast!
      with ThreadPoolExecutor(max_workers=get_worker_count(config)) as tp_exec:

        for epoch in range(config.epochs):
          # BEGIN TRAINING LOOP
          print("\n"*3)
          print("Training: {}/{}".format(epoch, config.epochs))
          overall_loss = 0
          sess.run(acc_init)  # clear accumulator for accuracy

          # process batches in parallel
          batch_futures = [tp_exec.submit(train_generator.__getitem__, i)
                           for i in range(len(train_generator))]
          batch_pbar = tqdm(as_completed(batch_futures),
                            total=len(train_generator))
          for count, batch_future in enumerate(batch_pbar):
            batch_data, batch_labels = batch_future.result()
            data = {input_placeholder: batch_data,
                    label_placeholder: batch_labels}

            _, l, _ = sess.run([optimizer,
                                loss,
                                acc_upop],
                               feed_dict=data)
            acc = sess.run(acc_watch)

            overall_loss += l
            per_batch_loss = overall_loss / (count+1)
            batch_pbar.set_description(
                "Loss {:0.5f} - Acc {:0.3f}".format(per_batch_loss, acc))

          # End of batch, log to tboard
          summary_writer.add_summary(sess.run(train_acc), epoch)
          train_generator.on_epoch_end()

          # VALIDATION LOOP
          print("Validation: {}/{}".format(epoch, config.epochs))
          sess.run(acc_init)
          overall_loss = 0
          batch_futures = [tp_exec.submit(val_generator.__getitem__, i)
                           for i in range(len(val_generator))]
          batch_pbar = tqdm(as_completed(batch_futures),
                            total=len(val_generator))
          for batch_future in batch_pbar:
            batch_data, batch_labels = batch_future.result()
            data = {input_placeholder: batch_data,
                   label_placeholder: batch_labels}
            l, _ = sess.run([loss, acc_upop], feed_dict=data)
            overall_loss += l
          print("Loss:", overall_loss / len(val_generator), "Acc:", sess.run(acc_watch))
          summary_writer.add_summary(sess.run(val_acc), epoch)
          val_generator.on_epoch_end()

  except Exception as e:
    raise e
  finally:
    print("Cleaning dirty tmp")
    tensorboard_dir.cleanup()


  return 0 # Exit Code
