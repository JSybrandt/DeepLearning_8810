#!/usr/bin/env python3

from scipy.io import loadmat
from cyberbully_detector.labels_pb2 import Annotation
from cyberbully_detector.labels_pb2 import DataClass
from cyberbully_detector.labels_pb2 import NO_BULLYING
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

annotation_dir = Path("./")
img_dir = Path("stanford_40")
for annotation_path in annotation_dir.glob("*.mat"):
  print(annotation_path)
  mat = loadmat(str(annotation_path), squeeze_me=True, struct_as_record=True)
  for raw_data in mat["annotation"]:
    annotation = Annotation()
    annotation.fold = fold_idx % num_folds
    fold_idx += 1
    annotation.file_path = str(img_dir.joinpath(str(raw_data["imageName"])))
    x1, y1, x2, y2 = raw_data["bbox"].tolist().tolist()

    annotation.dataset = "stanford_40"

    annotation_person = annotation.people.add()
    annotation_person.location.x = x1
    annotation_person.location.y = y1
    annotation_person.location.width = abs(x2-x1)
    annotation_person.location.height = abs(y2-y1)

    file_path = DATA_ROOT.joinpath(annotation.file_path)
    assert file_path.is_file()
    im = Image.open(str(file_path))
    width, height = im.size

    annotation.image_size.width = width
    annotation.image_size.height = height

    annotation.bullying_class = NO_BULLYING;

    assign_data_class(annotation)
    print(annotation)
    upload(annotation)
