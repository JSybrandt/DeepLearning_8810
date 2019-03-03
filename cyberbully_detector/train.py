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
from tensorflow.metrics import accuracy
from tensorflow.metrics import mean_squared_error as mse_watch

from tensorflow.train import Saver
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tempfile import TemporaryDirectory
import tensornets as nets

TRAINING_SCOPE="thawed"


def str_to_activation(activation_str):
  _activations = {
      "sigmoid": tf.nn.sigmoid,
      "relu": tf.nn.relu,
      "softmax": tf.nn.softmax,
      "tanh": tf.nn.tanh,
  }
  assert activation_str in _activations
  return _activations[activation_str]

def str_to_transfer_model(model_str):
  _models = {
      "resnet50"        : nets.ResNet50,
      "resnet101"       : nets.ResNet101,
      "resnet152"       : nets.ResNet152,
      "resnet50v2"      : nets.ResNet50v2,
      "resnet101v2"     : nets.ResNet101v2,
      "resnet152v2"     : nets.ResNet152v2,
      "resnet200v2"     : nets.ResNet200v2,
      "resnext50c32"    : nets.ResNeXt50c32,
      "resnext101c32"   : nets.ResNeXt101c32,
      "resnext101c64"   : nets.ResNeXt101c64,
      "wideresnet50"    : nets.WideResNet50,
      "inception1"      : nets.Inception1,
      "inception2"      : nets.Inception2,
      "inception3"      : nets.Inception3,
      "inception4"      : nets.Inception4,
      "inceptionresnet2": nets.InceptionResNet2,
      "nasnetalarge"    : nets.NASNetAlarge,
      "nasnetamobile"   : nets.NASNetAmobile,
      "pnasnetlarge"    : nets.PNASNetlarge,
      "vgg16"           : nets.VGG16,
      "vgg19"           : nets.VGG19,
      "densenet121"     : nets.DenseNet121,
      "densenet169"     : nets.DenseNet169,
      "densenet201"     : nets.DenseNet201,
      "mobilenet25"     : nets.MobileNet25,
      "mobilenet50"     : nets.MobileNet50,
      "mobilenet75"     : nets.MobileNet75,
      "mobilenet100"    : nets.MobileNet100,
      "mobilenet35v2"   : nets.MobileNet35v2,
      "mobilenet50v2"   : nets.MobileNet50v2,
      "mobilenet75v2"   : nets.MobileNet75v2,
      "mobilenet100v2"  : nets.MobileNet100v2,
      "mobilenet130v2"  : nets.MobileNet130v2,
      "mobilenet140v2"  : nets.MobileNet140v2,
      "squeezenet"      : nets.SqueezeNet,
  }
  assert model_str in _models
  return _models[model_str]

def initialize_model(config, num_people, current_layer):
  transfer_model = None
  for count, layer_conf in enumerate(config.model.layers):
    name = get_or_none(layer_conf, "name")

    if layer_conf.HasField("convolutional"):
      with tf.variable_scope(TRAINING_SCOPE):
        current_layer = Conv2D(
            layer_conf.convolutional.filters,
            (layer_conf.convolutional.kernel_size.width,
             layer_conf.convolutional.kernel_size.height),
            name=name,
            data_format="channels_first"
        )(current_layer)

    elif layer_conf.HasField("pool"):
      with tf.variable_scope(TRAINING_SCOPE):
        if layer_conf.pool.type == "max":
          current_layer = MaxPooling2D(
              (layer_conf.pool.size.width, layer_conf.pool.size.height),
              strides=(layer_conf.pool.size.width, layer_conf.pool.size.height),
              name=name
          )(current_layer)
        else:
          raise ValueError("Unsupported pool type:" + conv_config.pool_type)

    elif layer_conf.HasField("dense"):
      with tf.variable_scope(TRAINING_SCOPE):
        current_layer = Dense(
            layer_conf.dense.units,
            activation=str_to_activation(layer_conf.dense.activation),
            name=name
            )(current_layer)

    elif layer_conf.HasField("flatten"):
      with tf.variable_scope(TRAINING_SCOPE):
         current_layer = Flatten(name=name)(current_layer)

    elif layer_conf.HasField("dropout"):
      with tf.variable_scope(TRAINING_SCOPE):
        current_layer = Dropout(layer_conf.dropout.rate,
                                name=name)(current_layer)
    elif layer_conf.HasField("transfer"):
      with tf.variable_scope("transfer"):
        if count == 0:
          transfer_model = current_layer = str_to_transfer_model(
              layer_conf.name)(current_layer,
                               is_training=True,
                               stem=True)
        else:
          ValueError("Transfer layer must occur first.")
    else:
      ValueError("Unsupported layer.")

  return current_layer, transfer_model

def train_main(args):
  # Entry point into training from __main__.py

  config = get_config(args)
  if config.super_debug_mode:
    config.batch_size = 1

  if config.system.HasField("gpus"):
    raise ValueError("Multi-gpu support not yet impl")


  output_size = annotation_size(config.max_people_per_img);

  total_classes = enum_size(LB.BullyingClass)

  current_layer = input_placeholder = Input(
      shape=(None, # variable batch size
             config.target_size.width,
             config.target_size.height,
             colormode_to_dim(config.color_mode)),
      name="input",
      dtype=tf.float32)

  log.info("Creating a new model")
  prediction, transfer_model = initialize_model(config,
                                                  config.max_people_per_img,
                                                  current_layer)
  # CUSTOM END LEVEL

  bully_one_hot_placeholder = Input(shape=(None, total_classes),
                            dtype=tf.float32)

  # nan_mask = ~tf.is_nan(bully_one_hot_placeholder)
  # masked_label = tf.boolean_mask(bully_one_hot_placeholder, nan_mask)
  # masked_output = tf.boolean_mask(prediction, nan_mask)
  # loss = tf.losses.softmax_cross_entropy(masked_label, masked_output);

  loss = tf.losses.softmax_cross_entropy(bully_one_hot_placeholder, prediction)

  k_acc_watch, k_acc_up = accuracy(tf.argmax(bully_one_hot_placeholder, 1),
                                   tf.argmax(prediction, 1),
                                   name="k_acc_watch")

  k_acc_init = tf.variables_initializer(
      var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                 scope="k_acc_watch"))

  train_acc_k = tf.summary.scalar("Training_K-class_Acc", k_acc_watch)
  val_acc_k = tf.summary.scalar("Validation_K-class_Acc", k_acc_watch)

  thawed_vars = tf.trainable_variables(TRAINING_SCOPE)

  optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=thawed_vars)
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, var_list=thawed_vars)
  # optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.5).minimize(loss)
  #optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)

  # BELOW IS THE ACTUAL RUNNING

  train_generator = ImageAndAnnotationGenerator(
      data_path=args.data,
      data_class="TRAIN",
      num_people=config.max_people_per_img,
      sample_size=(config.target_size.width, config.target_size.height),
      short_side_size=get_or_none(config, "short_side_size"),
      batch_size=config.batch_size,
      datasets=args.dataset,
      extra_preproc_func=None if transfer_model is None else transfer_model.preprocess
  )
  val_generator = ImageAndAnnotationGenerator(
      data_path=args.data,
      data_class="VALIDATION",
      num_people=config.max_people_per_img,
      sample_size=(config.target_size.width, config.target_size.height),
      short_side_size=get_or_none(config, "short_side_size"),
      batch_size=config.batch_size,
      datasets=args.dataset,
      extra_preproc_func=None if transfer_model is None else transfer_model.preprocess
  )

  tensorboard_dir = TemporaryDirectory()
  saver = tf.train.Saver()

  num_trainable_param = np.sum([np.prod(v.get_shape().as_list())
                                for v in thawed_vars])

  log.info("Num trainable params: %s", num_trainable_param)

  try:
    with tf.Session() as sess:
      if transfer_model is not None:
        sess.run(transfer_model.pretrained())

    # ACTUALLY OPTIMIZE!
      log.info("Follow along with tensorboard: " + tensorboard_dir.name)
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
          sess.run(k_acc_init)  # clear accumulator for accuracy

          # process batches in parallel
          batch_futures = [tp_exec.submit(train_generator.__getitem__, i)
                           for i in range(len(train_generator))]
          batch_pbar = tqdm(as_completed(batch_futures),
                            total=len(train_generator))
          for count, batch_future in enumerate(batch_pbar):
            batch_data, batch_bully_class, batch_contains_bullying = batch_future.result()
            data = {input_placeholder: batch_data,
                    bully_one_hot_placeholder: batch_bully_class}
            l = sess.run([loss, optimizer,
                          k_acc_up],
                         feed_dict=data)[0]
            running_acc = sess.run(k_acc_watch)

            if config.super_debug_mode:
              print("Label:", batch_bully_class)
              print("Prediction:", sess.run(prediction, feed_dict=data))
              print("Loss:", l)
              print("Running Ac:", running_acc)
              input("Press Enter to continue")

            overall_loss += l
            if np.isnan(overall_loss):
              print(batch_data)
              print(batch_bully_class)
              raise RuntimeError("Nan Loss!")
            if overall_loss < 0:
              print(batch_data)
              print(batch_bully_class)
              raise RuntimeError("Negative Loss!")
            per_batch_loss = overall_loss / (count+1)
            batch_pbar.set_description(
                "Loss {:0.5f} - K-Acc {:0.3f}".format(per_batch_loss, running_acc))

          # End of batch, log to tboard
          summary_writer.add_summary(sess.run(train_acc_k), epoch)
          train_generator.on_epoch_end()

          # VALIDATION LOOP
          print("Validation: {}/{}".format(epoch, config.epochs))
          sess.run(k_acc_init)
          overall_loss = 0
          batch_futures = [tp_exec.submit(val_generator.__getitem__, i)
                           for i in range(len(val_generator))]
          batch_pbar = tqdm(as_completed(batch_futures),
                            total=len(val_generator))
          for batch_future in batch_pbar:
            batch_data, batch_bully_class, batch_contains_bullying = batch_future.result()
            data = {input_placeholder: batch_data,
                    bully_one_hot_placeholder: batch_bully_class}
            l = sess.run([loss, k_acc_up], feed_dict=data)[0]
            overall_loss += l
          log.info("Loss: %s K-Acc %s",  overall_loss / len(val_generator),
                                         sess.run(k_acc_watch))
          summary_writer.add_summary(sess.run(val_acc_k), epoch)
          val_generator.on_epoch_end()

          # save each epoch
          saver.save(sess, str(args.model.joinpath("model")))

  except Exception as e:
    raise e
  finally:
    print("Cleaning dirty tmp")
    tensorboard_dir.cleanup()


  return 0 # Exit Code
