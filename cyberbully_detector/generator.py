# This class is going to generate data from 
# Helpful info found here:
# https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a

from .data_util import get_annotation_ids
from .data_util import load_annotation
from .labels_pb2 import Annotation
from PIL import Image, ImageOps
import random
from pathlib import Path

from multiprocessing import Pool

class ImageAndAnnotationGenerator:
  def __init__(self, data_path, seed=None):
    """
    data_path: the root location of images. Inside should be a dir per dataset
    seed: set random seed
    """
    self.data_path = Path(data_path)
    if seed is not None:
      random.seed(seed)

  def _load_image(self, annotation):
    img_path = self.data_path.joinpath(annotation.file_path)
    assert img_path.is_file()
    return Image.open(str(img_path))

  def _resize_by_short_side(self, img, annotation):
    original_w, original_h = img.size
    min_size = min(original_w, original_h)
    rescale_ratio = self.short_side_size / min_size
    desired_w = max(int(original_w * rescale_ratio), self.short_side_size)
    desired_h = max(int(original_h * rescale_ratio), self.short_side_size)
    img = img.resize((desired_w, desired_h), Image.ANTIALIAS)

    for person in annotation.people:
      person.location.x *= rescale_ratio
      person.location.y *= rescale_ratio
      person.location.width *= rescale_ratio
      person.location.height *= rescale_ratio

    return img, annotation

  def _crop_rand_sample(self, img, annotation):
    assert self.sample_size[0] <= img.size[0]
    assert self.sample_size[1] <= img.size[1]

    # amt we can shift left-right
    x_diff = img.size[0] - self.sample_size[0]
    # amt we can shirt up-down
    y_diff = img.size[1] - self.sample_size[1]

    # shift random amt
    x_diff = int(x_diff * random.random())
    y_diff = int(y_diff * random.random())

    img = img.crop((x_diff,
                    y_diff,
                    x_diff+self.sample_size[0],
                    y_diff+self.sample_size[1]))

    annotation_cpy = Annotation()
    annotation_cpy.CopyFrom(annotation)
    annotation.ClearField("people")

    for person in annotation_cpy.people:
      if person.location.x > x_diff + self.sample_size[0]:
        continue
      if person.location.y > y_diff + self.sample_size[1]:
        continue
      if person.location.x + person.location.width < x_diff:
        continue
      if person.location.y + person.location.height < y_diff:
        continue

      person.location.x = person.location.x - x_diff
      person.location.y = person.location.y - y_diff
      if person.location.x < 0:
        person.location.width += person.location.x
        person.location.x = 0
      if person.location.y < 0:
        person.location.height += person.location.y
        person.location.y = 0

      annotation.people.add().CopyFrom(person)
    return img, annotation

  def _horizontal_flip(self, img, annotation):
    img = ImageOps.mirror(img)
    for person in annotation.people:
      person.location.x = img.size[0] - person.location.x - person.location.width
    return img, annotation

  def _vertical_flip(self, img, annotation):
    img = ImageOps.flip(img)
    for person in annotation.people:
      person.location.y = img.size[1] - person.location.y - person.location.height
    return img, annotation

  def _process_id(self, mongo_id):
    # Loads image from ref_annotation
    # Performs select augmentations
    # modify annotation in accordance
    annotation = load_annotation(mongo_id)
    img = self._load_image(annotation)

    img, annotation = self._resize_by_short_side(img, annotation)
    img, annotation = self._crop_rand_sample(img, annotation)
    if random.random() < 0.5:
      img, annotation = self._horizontal_flip(img, annotation)
    if random.random() < 0.5:
      img, annotation = self._vertical_flip(img, annotation)

    return img, annotation

  def flow_from_mongo(
      self,
      data_class,
      sample_size,
      short_side_size,
      batch_size,
      dataset=None,
      workers=1):
    """
    data_class: one of labels_pb2.DataClass
    sample_size: (width, height) in pixels of cropped image
    short_side_size: rescale image, preserves aspect ratio, such that the
                     shortest side is equal to short_side_size pixels
    batch_size: number of images
    dataset: subset to select
    """

    self.sample_size = sample_size
    self.short_side_size = short_side_size
    # Generators go forever
    while True:
      with Pool(workers) as pool:
        ids = get_annotation_ids(data_class, dataset)
        random.shuffle(ids)
        for batch_start in range(0, len(ids), batch_size):
          batch_ids = ids[batch_start:batch_start + batch_size]
          image_annotation_list = [self._process_id(id_) for id_ in batch_ids]
          #image_annotation_list = pool.map(self._process_id, batch_ids)
          images = [x[0] for x in image_annotation_list]
          annotations = [x[1] for x in image_annotation_list]
          yield images, annotations

