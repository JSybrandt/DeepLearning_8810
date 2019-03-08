#!/usr/bin/env python3

from scipy.io import loadmat
from cyberbully_detector.labels_pb2 import Annotation
from cyberbully_detector.labels_pb2 import DataClass
from cyberbully_detector.labels_pb2 import BullyingClass
from google.protobuf import json_format
from json import loads as json_loads
from pathlib import Path
from pymongo import MongoClient
from random import random
from PIL import Image

DATA_ROOT=Path("/home/jsybran/DeepLearning_8810/data")

MONGO_HOST = "jcloud"
mongo_client = MongoClient(MONGO_HOST)

def assign_data_class(annotation):
  rand = random()
  if rand < 0.2:
    annotation.data_class = DataClass.Value("TEST")
  elif rand < 0.3:
    annotation.data_class = DataClass.Value("VALIDATION")
  else:
    annotation.data_class = DataClass.Value("TRAIN")

def upload(annotation):
  d = json_loads(json_format.MessageToJson(annotation))
  mongo_client.DL_8810.annotations.insert_one(d)


num_folds = 3
fold_idx = 0

img_dir = DATA_ROOT.joinpath("bullying")
for class_dir in img_dir.glob("*"):
  if class_dir.is_dir():
    bully_class = BullyingClass.Value(class_dir.name.upper())
    for img_path in class_dir.glob("*"):
      if img_path.is_file():
        annotation = Annotation()
        annotation.fold = fold_idx % num_folds
        fold_idx += 1
        annotation.dataset = "bullying"
        annotation.file_path = str(Path("bullying").joinpath(class_dir.name).joinpath(img_path.name))
        annotation.bullying_class = bully_class

        im = Image.open(str(img_path))
        width, height = im.size
        annotation.image_size.width = width
        annotation.image_size.height = height

        assign_data_class(annotation)
        print(annotation)

        upload(annotation)
