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
from tensorflow.losses import mean_squared_error

from tensorflow.train import Saver
import numpy as np
from tqdm import tqdm

def str_to_activation(activation_str):
  _activations = {
      "sigmoid": tf.nn.sigmoid,
      "relu": tf.nn.relu,
      "softmax": tf.nn.softmax,
      "tanh": tf.nn.tanh,
  }
  assert activation_str in _activations
  return _activations[activation_str]

def get_output_layers(num_people, prev_layer):
  outputs = []
  # Contains bullying
  outputs.append(Dense(1,
                      activation=tf.nn.tanh,
                      name="contains_bullying")(prev_layer))
  # Bully Type
  outputs.append(Dense(enum_size(LB.BullyingClass),
                      activation=tf.nn.softmax,
                      name="bullying_class")(prev_layer))
  # For each person
  for p_idx in range(num_people):
    outputs.append(Dense(1,
                        activation=tf.nn.tanh,
                        name="exists_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(simple_proto_size(LB.Bbox),
                        activation=tf.nn.relu,
                        name="bbox_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.EmotionClass),
                        activation=tf.nn.sigmoid,
                        name="discrete_emotion_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(simple_proto_size(LB.ContinuousEmotion),
                        activation=tf.nn.relu,
                        name="continuous_emotion_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.Role),
                        activation=tf.nn.softmax,
                        name="role_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.Gender),
                        activation=tf.nn.softmax,
                        name="gender_{}".format(p_idx)
                       )(prev_layer))
    outputs.append(Dense(enum_size(LB.Age),
                        activation=tf.nn.softmax,
                        name="age_{}".format(p_idx)
                       )(prev_layer))
  return outputs

def add_convolutional(conv_config, current_layer, name):
  current_layer = Conv2D(
      conv_config.filters,
      (conv_config.kernel_size.width, conv_config.kernel_size.height),
      name=name
  )(current_layer)

  if name is not None:
    name += "_pool"

  if conv_config.pool_type == "max":
    current_layer = MaxPooling2D(
        (conv_config.pool_size.width, conv_config.pool_size.height),
        strides=(conv_config.pool_size.width, conv_config.pool_size.height),
        name=name
    )(current_layer)
  else:
    raise ValueError("Unsupported pool type:" + conv_config.pool_type)
  return current_layer

def initialize_model(config, num_people):
  current_layer = data_placeholder = Input(
      shape=(None, # variable batch size
             config.target_size.width,
             config.target_size.height,
             colormode_to_dim(config.color_mode)),
      dtype=tf.float32)

  # For each intermediate layer
  for count, layer_conf in enumerate(config.model.layers):
    name = get_or_none(layer_conf, "name")
    if layer_conf.HasField("convolutional"):
      current_layer = add_convolutional(layer_conf.convolutional,
                                        current_layer,
                                        name=name)
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

  outputs = get_output_layers(num_people, current_layer)
  return data_placeholder, outputs

def initialize_optimizer(config):
  opt_conf = config.model.optimizer
  if opt_conf.HasField("sgd"):
    return MomentumOptimizer(
               learning_rate=opt_conf.sgd.learning_rate,
               momentum=opt_conf.sgd.momentum,
               #nesterov=opt_conf.sgd.nesterov
               )
  else:
    raise ValueError("Must supply SGD optimizer.")

def train_main(args):
  # Entry point into training from __main__.py

  config = get_config(args)

  if config.system.HasField("gpus"):
    raise ValueError("Multi-gpu support not yet impl")

  split_output = False

  train_generator = ImageAndAnnotationGenerator(
      data_path=args.data,
      data_class="TRAIN",
      num_people=config.max_people_per_img,
      sample_size=(config.target_size.width, config.target_size.height),
      short_side_size=get_or_none(config, "short_side_size"),
      batch_size=config.batch_size,
      split_output=split_output,
      dataset=args.dataset
  )
  val_generator = ImageAndAnnotationGenerator(
      data_path=args.data,
      data_class="VALIDATION",
      num_people=config.max_people_per_img,
      sample_size=(config.target_size.width, config.target_size.height),
      short_side_size=get_or_none(config, "short_side_size"),
      batch_size=config.batch_size,
      split_output=split_output,
      dataset=args.dataset
  )

  #model_io = Saver()

  output_size = annotation_size(config.max_people_per_img);

  # setup model info
  #if args.model.is_file():
  #  log.info("Loading model to continue training from %s", args.model)
  #  model_io.restore(sess, args.model)
  #else:
  log.info("Creating a new model")
  data_placeholder, outputs = initialize_model(
      config, config.max_people_per_img)
  output = Concatenate(outputs, 1)
  label_placeholder = Input(shape=(None, output_size),
                            dtype=tf.float32)

  # If nan, don't compute
  nan_mask = tf.is_nan(label_placeholder)
  masked_label = tf.boolean_mask(label_placeholder, nan_mask)
  masked_output = tf.boolean_mask(output, nan_mask)

  cost = mean_squared_error(masked_label, masked_output);
  optimizer = initialize_optimizer(config).minimize(cost);

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(config.epochs):
      print(epoch, "/", config.epochs)
      for batch_data, batch_labels in tqdm(train_generator):
        sess.run([optimizer, cost], feed_dict={
          data_placeholder: batch_data,
          label_placeholder: batch_labels,
        })
      seq.on_epoch_end()

  return 0 # Exit Code
