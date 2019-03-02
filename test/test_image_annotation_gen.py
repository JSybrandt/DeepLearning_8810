#!/usr/bin/env python3

from cyberbully_detector.data_util import load_annotation
from cyberbully_detector.data_util import get_annotation_ids
from cyberbully_detector.generator import ImageAndAnnotationGenerator
from cyberbully_detector.visualize import draw_boxes_on_image
from cyberbully_detector.proto_util import vector_to_annotation
from cyberbully_detector.proto_util import unscale_people
from cyberbully_detector.visualize import np_array_to_img
from PIL import Image
import numpy as np

batch_size=100

gen = ImageAndAnnotationGenerator(
    "../data",
    "TRAIN",
    num_people=3,
    dataset="bullying",
    sample_size=(299,299),
    short_side_size=500,
    batch_size=batch_size)


data,labels = gen[0]

assert data.shape[0] == batch_size
joined_labels = np.hstack(labels)
assert joined_labels.shape[0] == batch_size

for idx in range(batch_size):

  label = joined_labels[idx, :]
  img_data = data[idx]

  img = np_array_to_img(img_data)
  annotation = vector_to_annotation(label)
  unscale_people(img, annotation)

  for person in annotation.people:
    print(person)
    assert person.location.x >= -0.0001
    assert person.location.y >= -0.0001
    assert person.location.width >= 0
    assert person.location.height >= 0
    assert person.location.x + person.location.width <= img.size[0] + 0.0001
    assert person.location.y + person.location.height <= img.size[1] + 0.0001
  if idx == 0:
    img = draw_boxes_on_image(img, annotation)
    img.show()
    break


for data, label in gen.generator(1):
  pass
