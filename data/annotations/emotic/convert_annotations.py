#!/usr/bin/env python3

ANNOTATION_FILE = "./Annotations.mat"

from scipy.io import loadmat
import numpy as np
from cyberbully_detector.labels_pb2 import Annotation
from cyberbully_detector.labels_pb2 import EmotionClass
from cyberbully_detector.labels_pb2 import Gender
from cyberbully_detector.labels_pb2 import Age
from cyberbully_detector.labels_pb2 import DataClass
from pathlib import Path
from pymongo import MongoClient
from google.protobuf import json_format
from json import loads as json_loads
from random import random


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

mat = loadmat(ANNOTATION_FILE, squeeze_me=True, struct_as_record=True)

def parse_people(people_data):
  fields = people_data.dtype.names
  # If only one person
  if people_data["gender"].shape == ():
    person = {
      "gender": str(people_data["gender"]),
      "age": str(people_data["age"]),
      "bbox": people_data["body_bbox"].tolist().tolist(),
    }

    if "combined_categories" in fields:
      categories = people_data["combined_categories"].tolist()
      if type(categories) == str:
        person["categories"] = [categories]
      else:
        person["categories"] = categories.tolist()
    else:
      categories = people_data["annotations_categories"].tolist()
      if type(categories) == str:
        person["categories"] = [categories]
      else:
        categories = categories["categories"].tolist()
        if type(categories) == str:
          person["categories"] = [categories]
        else:
          person["categories"] = categories.tolist()

    if "combined_continuous" in fields:
      comb_field = "combined_continuous"
    else:
      comb_field = "annotations_continuous"
    comb_data = people_data[comb_field].tolist()
    person["continuous"] = {emotion: float(comb_data[emotion]) for emotion in comb_data.dtype.names}
    yield person
  else:
    for person_idx in range(len(people_data["gender"])):
      person = {
        "gender": people_data["gender"][person_idx],
        "age": people_data["age"][person_idx],
        "bbox": people_data["body_bbox"][person_idx].tolist(),
      }

      if "combined_categories" in fields:
        categories = people_data["combined_categories"][person_idx]
        if type(categories) == str:
          person["categories"] = [categories]
        else:
          person["categories"] = categories.tolist()
      else: # annotations catregories
        categories = people_data["annotations_categories"][person_idx]
        if type(categories) == str:
          person["categories"] = [categories]
        else:
          categories = categories["categories"].tolist()
          if type(categories) == str:
            person["categories"] = [categories]
          else:
            person["categories"] = categories.tolist()

      if "combined_continuous" in fields:
        comb_field = "combined_continuous"
      else:
        comb_field = "annotations_continuous"
      comb_data = people_data[comb_field][person_idx]
      person["continuous"] = {emotion: float(comb_data[emotion]) for emotion in comb_data.dtype.names}

      yield person

num_folds = 3
fold_idx = 0

for part in ["val", "test", "train"]:
  for idx, data in enumerate(mat[part]):
    annotation = Annotation()
    annotation.fold = fold_idx % num_folds
    fold_idx += 1
    annotation.dataset = "emotic"
    annotation.file_path = str(Path("emotic").joinpath(data[1]).joinpath(data[0]))
    annotation.image_size.width = data[2]["n_col"]
    annotation.image_size.height = data[2]["n_row"]

    for raw_person in parse_people(data[4]):
      annotation_person = annotation.people.add()
      x1, y1, x2, y2 = raw_person["bbox"]
      annotation_person.location.x = x1
      annotation_person.location.y = y1
      annotation_person.location.width = abs(x2-x1)
      annotation_person.location.height = abs(y2-y1)

      annotation_person.gender = Gender.Value(raw_person["gender"].upper())
      annotation_person.age = Age.Value(raw_person["age"].upper())

      for emotion in raw_person["categories"]:
        annotation_person.discrete_emotion.emotions.append(EmotionClass.Value(emotion.upper().replace('/', '_')))

      annotation_person.continuous_emotion.valence = raw_person["continuous"]["valence"]
      annotation_person.continuous_emotion.arousal = raw_person["continuous"]["arousal"]
      annotation_person.continuous_emotion.dominance = raw_person["continuous"]["dominance"]

    assign_data_class(annotation)
    print(annotation)
    upload(annotation)

