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
from glob import iglob
import abc
from tqdm import tqdm
from . import labels_pb2 as LB
import logging as log

def _load_image(img_path):
  if img_path.is_file():
    return Image.open(str(img_path)).convert("RGB")
  else:
    raise RuntimeError("{} is not a file!".format(img_path))


def _resize(img, annotation, desired_w, desired_h):
  original_w, original_h = img.size
  img = img.resize((desired_w, desired_h), Image.ANTIALIAS)
  w_ratio = desired_w / original_w
  h_ratio = desired_h / original_h
  if annotation is not None:
    for person in annotation.people:
      if person.HasField("location"):
        person.location.x *= w_ratio
        person.location.y *= h_ratio
        person.location.width *= w_ratio
        person.location.height *= h_ratio

  return img, annotation

def _resize_by_short_side(img, annotation, short_side_size):
  original_w, original_h = img.size
  min_size = min(original_w, original_h)
  rescale_ratio = short_side_size / min_size
  desired_w = max(int(original_w * rescale_ratio), short_side_size)
  desired_h = max(int(original_h * rescale_ratio), short_side_size)
  return _resize(img, annotation, desired_w, desired_h)

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

  if annotation is not None:
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
  if annotation is not None:
    for person in annotation.people:
      if person.HasField("location"):
        person.location.x = img.size[0] - person.location.x - person.location.width
  return img, annotation

def _vertical_flip(img, annotation):
  img = ImageOps.flip(img)
  if annotation is not None:
    for person in annotation.people:
      if person.HasField("location"):
        person.location.y = img.size[1] - person.location.y - person.location.height
  return img, annotation

def _process_id(db, mongo_id, data_path, sample_size, short_side_size, num_people):
  # Loads image from ref_annotation
  # Performs select augmentations
  # modify annotation in accordance
  annotation = load_annotation(db, mongo_id)
  img_path = data_path.joinpath(annotation.file_path)

  img = _load_image(img_path)

  if short_side_size is None:
    img, annotation = _resize(img, annotation, sample_size[0], sample_size[0])
  else:
    assert short_side_size >= max(sample_size)
    img, annotation = _resize_by_short_side(img, annotation, short_side_size)
    img, annotation = _crop_rand_sample(img, annotation, sample_size)

  if random.random() < 0.5:
    img, annotation = _horizontal_flip(img, annotation)
  if random.random() < 0.5:
    img, annotation = _vertical_flip(img, annotation)

  zero_one_scale_people(img, annotation)

  arr = np.array(img, dtype=np.float32)
  vector = annotation_to_vector(annotation, num_people)
  contains_bullying = 0 if annotation.bullying_class == LB.NO_BULLYING else 1
  return arr, vector, contains_bullying

def proc_img_path(path, short_side_size, sample_size, callbacks=[], flips=False, force_horizontal_flip=False, force_vertical_flip=False):
  img = _load_image(path)

  if short_side_size is None:
    img, _ = _resize(img, None, sample_size[0], sample_size[0])
  else:
    assert short_side_size >= max(sample_size)
    img, _ = _resize_by_short_side(img, None, short_side_size)
    img, _ = _crop_rand_sample(img, None, sample_size)

  if flips:
    if random.random() < 0.5:
      img, _ = _horizontal_flip(img, None)
    if random.random() < 0.5:
      img, _ = _vertical_flip(img, None)
  else:
    if force_horizontal_flip:
      img, _ = _horizontal_flip(img, None)
    if force_vertical_flip:
      img, _ = _vertical_flip(img, None)

  for callback in callbacks:
    callback(path, img)
  arr = np.array(img, dtype=np.float32)
  return arr


class Sequence(metaclass=abc.ABCMeta):
  # 100% BASED ON KERAS SEQUENCE CLASS
  # Didn't copy, just want the interface

  def __iter__(self):
    for batch_idx in range(len(self)):
      yield self[batch_idx]

  @abc.abstractmethod
  def __len__(self):
    pass

  @abc.abstractmethod
  def on_epoch_end(self):
    pass

  @abc.abstractmethod
  def __getitem__(self, batch_idx):
    pass


class ImageAndAnnotationGenerator(Sequence):
  def __init__(self,
               num_people,
               sample_size,
               batch_size,
               data_class=None,
               folds=None,  # Array of folds #'s
               data_path=None,
               short_side_size=None,
               datasets=None,
               seed=None,
               extra_preproc_func=None,
               balance_classes=True):
    db = get_annotation_db_connection()
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

    self.ids = get_annotation_ids(db, data_class, datasets, folds)
    if balance_classes:
      log.info("Loading annotation classes for balancing.")
      self.class2ids = {}
      for _id in self.ids:
        annotation = load_annotation(self.db, _id)
        c = annotation.bullying_class
        if c in self.class2ids:
          self.class2ids[c].append(_id)
        else:
          self.class2ids[c] = [_id]
    else:
      # Put them all in one class
      self.class2ids = {-1: self.ids}
    self.classes = list(self.class2ids.keys())
    self.folds=folds
    self.balance_classes=balance_classes
    self.data_path = data_path
    self.num_people = num_people
    self.sample_size = sample_size
    self.batch_size = batch_size
    self.short_side_size = short_side_size
    self.extra_preproc_func = extra_preproc_func
    self.on_epoch_end()

  def __len__(self):
    # Number of batches
    # Largest class
    max_ids = max([len(i) for _, i in self.class2ids.items()])
    # div across classes (remember, only 1 class if not balance classes)
    batch_per_ds = self.batch_size / len(self.class2ids)
    # number of batches
    return int(np.ceil(max_ids/batch_per_ds))

  def on_epoch_end(self):
    for _, ids in self.class2ids.items():
      random.shuffle(ids)

  def __getitem__(self, batch_idx):
    data = np.empty((self.batch_size, self.sample_size[0], self.sample_size[1], 3))
    bully_class = np.empty((self.batch_size, annotation_size(self.num_people)))
    contains_bullying = np.empty((self.batch_size, 1))

    for i in range(self.batch_size):
      class_idx = random.choice(self.classes)
      annotation_id = random.choice(self.class2ids[class_idx])

      data[i,:,:,:], bully_class[i, :], contains_bullying[i, :] = _process_id(
          self.db,
          annotation_id,
          self.data_path,
          self.sample_size,
          self.short_side_size,
          self.num_people)
    data = self.extra_preproc_func(data)
    return data, bully_class, contains_bullying


class FileSystemImageGenerator(Sequence):
  def __init__(self,
               data_path,
               sample_size,
               batch_size,
               short_side_size=None,
               seed=None,
               img_callbacks=[],
               extra_preproc_func=None):

    assert len(sample_size) == 2
    assert min(sample_size) > 0
    self.sample_size = sample_size
    assert batch_size > 0
    self.batch_size = batch_size
    if short_side_size is None:
      self.short_side_size = max(sample_size)
    else:
      assert short_side_size >= max(sample_size)
      self.short_side_size = short_side_size
    if seed is not None:
      random.seed(seed)

    self.img_callbacks = img_callbacks
    self.extra_preproc_func = extra_preproc_func

    image_extensions = set([".png", ".jpg", ".jpeg"])
    files = iglob(str(data_path)+"/**", recursive=True)
    self.files = []
    for file_str in files:
      file_path = Path(file_str)
      assert file_path.exists()
      if file_path.is_file() and file_path.suffix.lower() in image_extensions:
        self.files.append(file_path)

  def get_files(self):
    return self.files

  def __len__(self):
    return int(np.ceil(len(self.files)/self.batch_size))

  def __getitem__(self, batch_idx):
    start_idx = batch_idx * self.batch_size
    end_idx = min(len(self.files), start_idx + self.batch_size)

    this_batch_size = end_idx - start_idx
    data = np.empty((this_batch_size, self.sample_size[0], self.sample_size[1], 3))
    for i in range(this_batch_size):
      idx = start_idx + i
      data[i,:,:,:] = proc_img_path(files[idx],
                                     self.short_side_size,
                                     self.sample_size,
                                     self.callbacks)

    data = selfaextra_preproc_func(data)
    return data

