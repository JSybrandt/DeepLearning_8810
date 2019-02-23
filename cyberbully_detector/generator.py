# This class is going to generate data from 
# Helpful info found here:
# https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a

from .data_util import get_annotation_ids
from .data_util import load_annotation
from .data_util import get_annotation_db_connection
from .labels_pb2 import Annotation
from .proto_util import annotation_to_vector
from .proto_util import annotation_size
from .proto_util import split_batch
from .proto_util import zero_one_scale_people
from .visualize import image_to_np_array
from PIL import Image, ImageOps
import random
from pathlib import Path
import numpy as np
import threading
from keras.utils import Sequence

def _load_image(data_path, annotation):
  img_path = data_path.joinpath(annotation.file_path)
  assert img_path.is_file()
  return Image.open(str(img_path))

def _resize_by_short_side(img, annotation, short_side_size):
  original_w, original_h = img.size
  min_size = min(original_w, original_h)
  rescale_ratio = short_side_size / min_size
  desired_w = max(int(original_w * rescale_ratio), short_side_size)
  desired_h = max(int(original_h * rescale_ratio), short_side_size)
  img = img.resize((desired_w, desired_h), Image.ANTIALIAS)

  for person in annotation.people:
    if person.HasField("location"):
      person.location.x *= rescale_ratio
      person.location.y *= rescale_ratio
      person.location.width *= rescale_ratio
      person.location.height *= rescale_ratio

  return img, annotation

def _crop_rand_sample(img, annotation, sample_size):
  assert sample_size[0] <= img.size[0]
  assert sample_size[1] <= img.size[1]

  # amt we can shift left-right
  x_diff = img.size[0] - sample_size[0]
  # amt we can shirt up-down
  y_diff = img.size[1] - sample_size[1]

  # shift random amt
  x_diff = int(x_diff * random.random())
  y_diff = int(y_diff * random.random())

  img = img.crop((x_diff,
                  y_diff,
                  x_diff+sample_size[0],
                  y_diff+sample_size[1]))
  assert img.size == sample_size

  annotation_cpy = Annotation()
  annotation_cpy.CopyFrom(annotation)
  annotation.ClearField("people")

  for person in annotation_cpy.people:
    if person.HasField("location"):
      if person.location.x > x_diff + sample_size[0]:
        continue
      if person.location.y > y_diff + sample_size[1]:
        continue
      if person.location.x + person.location.width < x_diff:
        continue
      if person.location.y + person.location.height < y_diff:
        continue
      person.location.x = person.location.x - x_diff
      person.location.y = person.location.y - y_diff
      if person.location.x < 0:
        # x is neg
        person.location.width += person.location.x
        person.location.x = 0
      if person.location.y < 0:
        # y is neg
        person.location.height += person.location.y
        person.location.y = 0

      # if bbox extends beyond img
      if person.location.width + person.location.x > sample_size[0]:
        person.location.width = sample_size[0] - person.location.x
      if person.location.height + person.location.y > sample_size[1]:
        person.location.height = sample_size[1] - person.location.y

      assert person.location.width <= img.size[0]
      assert person.location.height <= img.size[1]
      if person.location.height > 0 and person.location.width > 0:
        annotation.people.add().CopyFrom(person)
    else:
      annotation.people.add().CopyFrom(person)
  return img, annotation

def _horizontal_flip(img, annotation):
  img = ImageOps.mirror(img)
  for person in annotation.people:
    if person.HasField("location"):
      person.location.x = img.size[0] - person.location.x - person.location.width
  return img, annotation

def _vertical_flip(img, annotation):
  img = ImageOps.flip(img)
  for person in annotation.people:
    if person.HasField("location"):
      person.location.y = img.size[1] - person.location.y - person.location.height
  return img, annotation

def _process_id(db, mongo_id, data_path, sample_size, short_side_size, num_people):
  # Loads image from ref_annotation
  # Performs select augmentations
  # modify annotation in accordance
  annotation = load_annotation(db, mongo_id)
  img = _load_image(data_path, annotation)

  img, annotation = _resize_by_short_side(img, annotation, short_side_size)
  img, annotation = _crop_rand_sample(img, annotation, sample_size)
  if random.random() < 0.5:
    img, annotation = _horizontal_flip(img, annotation)
  if random.random() < 0.5:
    img, annotation = _vertical_flip(img, annotation)

  zero_one_scale_people(img, annotation)

  raw_data = image_to_np_array(img)
  vector = annotation_to_vector(annotation, num_people)
  return raw_data, vector


class ImageAndAnnotationGenerator(Sequence):
  def __init__(self,
               data_path,
               data_class,
               num_people,
               sample_size,
               batch_size,
               short_side_size=None,
               dataset=None,
               seed=None,
               split_output=True,
               multithreaded=True):

    db = get_annotation_db_connection()
    if multithreaded:
      self.db = db
    data_path = Path(data_path)
    assert data_path.exists()
    assert num_people >= 0
    assert len(sample_size) == 2
    assert min(sample_size) > 0
    assert batch_size > 0
    if short_side_size is None:
      short_side_size = max(sample_size)
    assert short_side_size >= max(sample_size)
    if seed is not None:
      random.seed(seed)

    self.ids = get_annotation_ids(db, data_class, dataset)
    self.data_path = data_path
    self.num_people = num_people
    self.sample_size = sample_size
    self.batch_size = batch_size
    self.short_side_size = short_side_size
    self.split_output = split_output
    self.multithreaded = multithreaded
    self.on_epoch_end()

  def __len__(self):
    # Number of batches
    return int(np.ceil(len(self.ids)/self.batch_size))

  def on_epoch_end(self):
    random.shuffle(self.ids)

  def __getitem__(self, batch_idx):
    if self.multithreaded:
      db = self.db
    else: # if using multi-process, we need to make a new db connection
      db = get_annotation_db_connection()

    start_idx = batch_idx * self.batch_size
    end_idx = min(len(self.ids), start_idx + self.batch_size)

    this_batch_size = end_idx - start_idx
    data = np.empty((this_batch_size, self.sample_size[0], self.sample_size[1], 3))
    labels = np.empty((this_batch_size, annotation_size(self.num_people)))
    for i in range(this_batch_size):
      idx = start_idx + i
      data[i,:,:,:], labels[i, :] = _process_id(db,
                                                self.ids[idx],
                                                self.data_path,
                                                self.sample_size,
                                                self.short_side_size,
                                                self.num_people)
    if self.split_output:
      return data, split_batch(labels)
    else:
      return data, labels

