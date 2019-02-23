#!/usr/bin/env python3

from cyberbully_detector.generator import ImageAndAnnotationGenerator
import cyberbully_detector.train as cb_train
from cyberbully_detector.proto_util import annotation_size
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D

num_people = 3
gen = ImageAndAnnotationGenerator(
    "../data",
    "TRAIN",
    num_people,
    dataset="emotic",
    sample_size=(255, 255),
    short_side_size=300,
    batch_size=32,
    split_output=False,
    multithreaded=False
    )

output_size = annotation_size(num_people)

input_layer = Input((255, 255, 3))
p1 = MaxPooling2D((2, 2))(input_layer) # 127
p2 = MaxPooling2D((2, 2))(p1) # 63
p3 = MaxPooling2D((2, 2))(p2) # 31
d1 = Flatten()(p3)
out = Dense(output_size)(d1)
model = Model(input_layer, out)
model.compile(optimizer="sgd", loss=cb_train.mse_nan)
model.fit_generator(
    gen,
    epochs=10,
    workers=39,
    max_queue_size=100,
    use_multiprocessing=True
    )
