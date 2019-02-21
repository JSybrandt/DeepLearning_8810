#!/usr/bin/env python3

from cyberbully_detector.data_util import load_annotation
from cyberbully_detector.data_util import get_annotation_ids
from cyberbully_detector.proto_util import annotation_to_vector
from cyberbully_detector.proto_util import vector_to_annotation
from cyberbully_detector.proto_util import enum_to_vec
from cyberbully_detector.proto_util import person_to_vec
from cyberbully_detector.proto_util import discrete_emotion_to_vec
from cyberbully_detector.proto_util import simple_proto_to_vec
import cyberbully_detector.labels_pb2 as LB
import numpy as np

def test_end_to_end():
  ids = get_annotation_ids("TRAIN")
  example_label = load_annotation(ids[0])
  example_label.ClearField('file_path')
  example_label.ClearField('dataset')
  example_label.ClearField('image_size')
  example_label.ClearField('data_class')

  example_vector = annotation_to_vector(example_label,2)

  recreated = vector_to_annotation(example_vector)

  print(example_label)
  print("---")
  print(example_vector)
  print("---")
  print(recreated)

def test_end_to_end_2():
  ids = get_annotation_ids("TRAIN", dataset="bullying")
  example_label = load_annotation(ids[0])
  example_label.ClearField('file_path')
  example_label.ClearField('dataset')
  example_label.ClearField('image_size')
  example_label.ClearField('data_class')

  example_vector = annotation_to_vector(example_label,2)

  recreated = vector_to_annotation(example_vector)

  print(example_label)
  print("---")
  print(example_vector)
  print("---")
  print(recreated)
  assert example_label == recreated

def test_enum():
  val = enum_to_vec(LB.TRAIN, LB.DataClass)
  assert len(val) == 3
  assert val[0] == 1
  assert val[1] == 0
  assert val[2] == 0

def test_discrete_emotions():
  d_em = LB.DiscreteEmotion()
  d_em.emotions.append(LB.PEACE)
  d_em.emotions.append(LB.AFFECTION)
  d_em.emotions.append(LB.ESTEEM)

  actual_val = discrete_emotion_to_vec(d_em)
  exp_val = np.zeros(27)
  exp_val[LB.PEACE-1] = 1
  exp_val[LB.AFFECTION-1] = 1
  exp_val[LB.ESTEEM-1] = 1

  assert np.array_equal(actual_val, exp_val)


def test_continuous_emotion():
  c_em = LB.ContinuousEmotion()
  c_em.valence = 1
  c_em.arousal = 2
  c_em.dominance = 3

  actual_val = simple_proto_to_vec(c_em)
  exp_val = [1, 2, 3]
  assert np.array_equal(actual_val, exp_val)

def test_person():
  person = LB.Person()
  person.location.x = 1
  person.location.y = 2
  person.location.width = 3
  person.location.height = 4
  person.discrete_emotion.emotions.append(LB.PEACE)
  person.discrete_emotion.emotions.append(LB.AFFECTION)
  person.discrete_emotion.emotions.append(LB.ESTEEM)

  person_vec = person_to_vec(person)
  assert person_vec[0] == 1 # I exist!
  assert person_vec[1] == 1 # x
  assert person_vec[2] == 2 # y
  assert person_vec[3] == 3 # width
  assert person_vec[4] == 4 # height
  assert person_vec[5] == 1 # Peace
  assert person_vec[6] == 1 # Affection
  assert person_vec[7] == 1 # Esteem
