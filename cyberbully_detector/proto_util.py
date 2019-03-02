from google.protobuf import json_format
from google.protobuf import text_format
from .labels_pb2 import Annotation
from . import labels_pb2 as LB
import json
import numpy as np
from random import sample
from random import shuffle

def parse_pb_to_proto(path, proto_obj):
  with open(path, 'rb') as proto_file:
    proto_obj.ParseFromString(proto_file.read())

def parse_json_to_proto(path, proto_obj):
  with open(path, 'r') as json_file:
    json_format.Parse(json_file.read(), proto_obj)

def parse_txt_to_proto(path, proto_obj):
  with open(path, 'r') as txt_file:
    text_format.Merge(txt_file.read(), proto_obj)

def get_or_none(proto, field_name):
  if proto.HasField(field_name):
    return getattr(proto, field_name)
  else:
    return None

PROTO_PARSERS = {
    ".pb": parse_pb_to_proto,
    ".json": parse_json_to_proto,
    ".txt": parse_txt_to_proto,
    ".proto": parse_txt_to_proto,
    ".config": parse_txt_to_proto,
}

def enum_size(enum_type):
  return max(enum_type.values())

def enum_to_vec(enum_val, enum_type):
  vals = np.empty(enum_size(enum_type))
  if enum_val is None:
    vals[:] = np.nan
  else:
    if type(enum_val) is not int:
      enum_val = enum.Value(enum_val)
    vals[:] = 0
    vals[enum_val-1] = 1
  return vals

def simple_proto_size(msg_type):
  return len(msg_type.DESCRIPTOR.fields)

def simple_proto_to_vec(msg):
  vals = np.empty(simple_proto_size(msg))
  for i, field in enumerate(msg.DESCRIPTOR.fields):
    if msg.HasField(field.name):
      vals[i] = getattr(msg, field.name)
    else:
      vals[i] = np.nan
  return vals

def discrete_emotion_size():
  return enum_size(LB.EmotionClass)

def discrete_emotion_to_vec(discrete_emotion):
  vals = np.empty(enum_size(LB.EmotionClass))
  if discrete_emotion is None:
    vals[:] = np.nan
  else:
    vals[:] = 0
    for emotion in discrete_emotion.emotions:
      vals[emotion-1] = 1
  return vals

def person_size():
  return sum([
    1, # Do I Exist?
    simple_proto_size(LB.Bbox),
    discrete_emotion_size(),
    simple_proto_size(LB.ContinuousEmotion),
    enum_size(LB.Role),
    enum_size(LB.Gender),
    enum_size(LB.Age)
  ])

def person_to_vec(person):
  if person is not None:
    return np.concatenate([
        [1],  # I exist!
        simple_proto_to_vec(person.location),
        discrete_emotion_to_vec(get_or_none(person, "discrete_emotion")),
        simple_proto_to_vec(person.continuous_emotion),
        enum_to_vec(get_or_none(person, "role"),
                    LB.Role),
        enum_to_vec(get_or_none(person, "gender"),
                    LB.Gender),
        enum_to_vec(get_or_none(person, "age"),
                    LB.Age)], axis=0)
  else:
    vals = np.empty(person_size())
    vals[0] = 0 # I do not exist
    vals[1:] = np.nan
    return vals

def bool_to_vec(bool_val):
  vals = np.empty(1)
  if bool_val is None:
    vals[0] = np.nan
  elif bool_val:
    vals[0] = 1
  else:
    vals[0] = 0
  return vals

def annotation_size(num_people):
  return sum([
    enum_size(LB.BullyingClass),
    person_size() * num_people
  ])

def annotation_to_vector(annotation, num_people):
  people = [person for person in annotation.people]
  while len(people) < num_people:
    people.append(None)
  if len(people) > num_people:
    people = sample(people, num_people)

  assert len(people) == num_people
  shuffle(people)

  return np.concatenate([
    enum_to_vec(get_or_none(annotation, "bullying_class"), LB.BullyingClass),
  ] + [person_to_vec(p) for p in people])

def split_batch(vectors):
  # converts a batch of vectors to multiple columns
  matrix = np.vstack(vectors)
  outputs = []
  col_idx = 0

  size=enum_size(LB.BullyingClass)
  outputs.append(matrix[:,col_idx:col_idx+size])
  col_idx+=size

  while col_idx < matrix.shape[1]:
    # exists
    size=1
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size
    size=simple_proto_size(LB.Bbox)
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size
    size=discrete_emotion_size()
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size
    size=simple_proto_size(LB.ContinuousEmotion)
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size
    size=enum_size(LB.Role)
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size
    size=enum_size(LB.Gender)
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size
    size=enum_size(LB.Age)
    outputs.append(matrix[:,col_idx:col_idx+size])
    col_idx+=size

  return outputs

def zero_one_scale_people(ref_img, annotation):
  for person in annotation.people:
    if person.HasField("location"):
      person.location.x /= ref_img.size[0]
      person.location.y /= ref_img.size[1]
      person.location.width /= ref_img.size[0]
      person.location.height /= ref_img.size[1]
    if person.HasField("continuous_emotion"):
      person.continuous_emotion.valence /= 10
      person.continuous_emotion.arousal /= 10
      person.continuous_emotion.dominance /= 10

# CONVERTING BACK

def val_to_bool(val):
  if np.isnan(val):
    return None
  return val > 0.5

def val_to_enum(vals):
  potential_idx = np.argmax(vals)
  if vals[potential_idx] > 0.5:
    return potential_idx+1
  return None

def vec_to_simple_proto(vals, msg):
  assert len(msg.DESCRIPTOR.fields) == len(vals)
  for i, field in enumerate(msg.DESCRIPTOR.fields):
    setattr(msg, field.name, vals[i])

def vec_to_discrete_emotion(vals, msg):
  assert enum_size(LB.EmotionClass) == len(vals)
  for idx, val in enumerate(vals):
    if val > 0.5:
      msg.emotions.append(idx+1)

def vector_to_person(vector):
  if vector[0] < 0.5:
    return None
  person = LB.Person()

  idx = 1
  size = simple_proto_size(person.location)
  vec = vector[idx:idx+size] ; idx += size
  vec_to_simple_proto(vec, person.location)

  size = discrete_emotion_size()
  vec = vector[idx:idx+size] ; idx += size
  vec_to_discrete_emotion(vec, person.discrete_emotion)

  size = simple_proto_size(person.continuous_emotion)
  vec = vector[idx:idx+size] ; idx += size
  vec_to_simple_proto(vec, person.continuous_emotion)

  size = enum_size(LB.Role)
  vec = vector[idx:idx+size] ; idx += size
  tmp = val_to_enum(vec)
  if tmp is not None:
    person.role = tmp

  size = enum_size(LB.Gender)
  vec = vector[idx:idx+size] ; idx += size
  tmp = val_to_enum(vec)
  if tmp is not None:
    person.gender = tmp

  size = enum_size(LB.Age)
  vec = vector[idx:idx+size] ; idx += size
  tmp = val_to_enum(vec)
  if tmp is not None:
    person.age = tmp
  return person


def vector_to_annotation(vector):
  annotation = Annotation()

  idx = 0

  num_vals = enum_size(LB.BullyingClass)
  tmp = val_to_enum(vector[idx:idx+num_vals])
  if tmp is not None:
    annotation.bullying_class = tmp
  idx += num_vals

  size = person_size()
  while idx < len(vector):
    person_vec = vector[idx:idx+size]
    tmp = vector_to_person(person_vec)
    if tmp is not None:
      person = annotation.people.add()
      person.CopyFrom(tmp)
    idx+=size

  assert idx == len(vector)
  return annotation

def unscale_people(ref_img, annotation):
  for person in annotation.people:
    if person.HasField("location"):
      person.location.x *= ref_img.size[0]
      person.location.y *= ref_img.size[1]
      person.location.width *= ref_img.size[0]
      person.location.height *= ref_img.size[1]
    if person.HasField("continuous_emotion"):
      person.continuous_emotion.valence *= 10
      person.continuous_emotion.arousal *= 10
      person.continuous_emotion.dominance *= 10
