# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: labels.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='labels.proto',
  package='cyberbully_detector',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x0clabels.proto\x12\x13\x63yberbully_detector\x1a\x0c\x63ommon.proto\";\n\x04\x42\x62ox\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\r\n\x05width\x18\x03 \x01(\x02\x12\x0e\n\x06height\x18\x04 \x01(\x02\"F\n\x0f\x44iscreteEmotion\x12\x33\n\x08\x65motions\x18\x01 \x03(\x0e\x32!.cyberbully_detector.EmotionClass\"H\n\x11\x43ontinuousEmotion\x12\x0f\n\x07valence\x18\x01 \x01(\x02\x12\x0f\n\x07\x61rousal\x18\x02 \x01(\x02\x12\x11\n\tdominance\x18\x03 \x01(\x02\"\xb6\x02\n\x06Person\x12+\n\x08location\x18\x01 \x01(\x0b\x32\x19.cyberbully_detector.Bbox\x12>\n\x10\x64iscrete_emotion\x18\x02 \x01(\x0b\x32$.cyberbully_detector.DiscreteEmotion\x12\x42\n\x12\x63ontinuous_emotion\x18\x03 \x01(\x0b\x32&.cyberbully_detector.ContinuousEmotion\x12\'\n\x04role\x18\x04 \x01(\x0e\x32\x19.cyberbully_detector.Role\x12+\n\x06gender\x18\x05 \x01(\x0e\x32\x1b.cyberbully_detector.Gender\x12%\n\x03\x61ge\x18\x06 \x01(\x0e\x32\x18.cyberbully_detector.Age\"\x8a\x02\n\nAnnotation\x12:\n\x0e\x62ullying_class\x18\x01 \x01(\x0e\x32\".cyberbully_detector.BullyingClass\x12+\n\x06people\x18\x03 \x03(\x0b\x32\x1b.cyberbully_detector.Person\x12\x11\n\tfile_path\x18\x04 \x01(\t\x12\x0f\n\x07\x64\x61taset\x18\x05 \x01(\t\x12-\n\nimage_size\x18\x06 \x01(\x0b\x32\x19.cyberbully_detector.Size\x12\x32\n\ndata_class\x18\x07 \x01(\x0e\x32\x1e.cyberbully_detector.DataClass\x12\x0c\n\x04\x66old\x18\x08 \x01(\x05*\xa0\x03\n\x0c\x45motionClass\x12\t\n\x05PEACE\x10\x01\x12\r\n\tAFFECTION\x10\x02\x12\n\n\x06\x45STEEM\x10\x03\x12\x10\n\x0c\x41NTICIPATION\x10\x04\x12\x0e\n\nENGAGEMENT\x10\x05\x12\x0e\n\nCONFIDENCE\x10\x06\x12\r\n\tHAPPINESS\x10\x07\x12\x0c\n\x08PLEASURE\x10\x08\x12\x0e\n\nEXCITEMENT\x10\t\x12\x0c\n\x08SURPRISE\x10\n\x12\x0c\n\x08SYMPATHY\x10\x0b\x12\r\n\tCONFUSION\x10\x0c\x12\x11\n\rDISCONNECTION\x10\r\x12\x0b\n\x07\x46\x41TIGUE\x10\x0e\x12\x11\n\rEMBARRASSMENT\x10\x0f\x12\x0c\n\x08YEARNING\x10\x10\x12\x0f\n\x0b\x44ISAPPROVAL\x10\x11\x12\x0c\n\x08\x41VERSION\x10\x12\x12\r\n\tANNOYANCE\x10\x13\x12\t\n\x05\x41NGER\x10\x14\x12\x0f\n\x0bSENSITIVITY\x10\x15\x12\x0b\n\x07SADNESS\x10\x16\x12\x10\n\x0c\x44ISQUIETMENT\x10\x17\x12\x08\n\x04\x46\x45\x41R\x10\x18\x12\x08\n\x04PAIN\x10\x19\x12\r\n\tSUFFERING\x10\x1a\x12\x13\n\x0f\x44OUBT_CONFUSION\x10\x1b*,\n\x04Role\x12\t\n\x05\x42ULLY\x10\x01\x12\n\n\x06VICTIM\x10\x02\x12\r\n\tBYSTANDER\x10\x03*\x1e\n\x06Gender\x12\x08\n\x04MALE\x10\x01\x12\n\n\x06\x46\x45MALE\x10\x02*\'\n\x03\x41ge\x12\x07\n\x03KID\x10\x01\x12\x0c\n\x08TEENAGER\x10\x02\x12\t\n\x05\x41\x44ULT\x10\x03*\xa8\x01\n\rBullyingClass\x12\r\n\tGOSSIPING\x10\x01\x12\x0c\n\x08LAUGHING\x10\x02\x12\x10\n\x0cPULLING_HAIR\x10\x03\x12\x0e\n\nQUARRELING\x10\x04\x12\x0c\n\x08STABBING\x10\x05\x12\r\n\tISOLATION\x10\x06\x12\x0c\n\x08PUNCHING\x10\x07\x12\x0c\n\x08SLAPPING\x10\x08\x12\x0e\n\nSTRANGLING\x10\t\x12\x0f\n\x0bNO_BULLYING\x10\n*0\n\tDataClass\x12\t\n\x05TRAIN\x10\x01\x12\x08\n\x04TEST\x10\x02\x12\x0e\n\nVALIDATION\x10\x03')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])

_EMOTIONCLASS = _descriptor.EnumDescriptor(
  name='EmotionClass',
  full_name='cyberbully_detector.EmotionClass',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PEACE', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AFFECTION', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ESTEEM', index=2, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ANTICIPATION', index=3, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ENGAGEMENT', index=4, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONFIDENCE', index=5, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HAPPINESS', index=6, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PLEASURE', index=7, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EXCITEMENT', index=8, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SURPRISE', index=9, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SYMPATHY', index=10, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONFUSION', index=11, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DISCONNECTION', index=12, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FATIGUE', index=13, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EMBARRASSMENT', index=14, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='YEARNING', index=15, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DISAPPROVAL', index=16, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AVERSION', index=17, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ANNOYANCE', index=18, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ANGER', index=19, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SENSITIVITY', index=20, number=21,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SADNESS', index=21, number=22,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DISQUIETMENT', index=22, number=23,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FEAR', index=23, number=24,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PAIN', index=24, number=25,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUFFERING', index=25, number=26,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOUBT_CONFUSION', index=26, number=27,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=841,
  serialized_end=1257,
)
_sym_db.RegisterEnumDescriptor(_EMOTIONCLASS)

EmotionClass = enum_type_wrapper.EnumTypeWrapper(_EMOTIONCLASS)
_ROLE = _descriptor.EnumDescriptor(
  name='Role',
  full_name='cyberbully_detector.Role',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BULLY', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VICTIM', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BYSTANDER', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1259,
  serialized_end=1303,
)
_sym_db.RegisterEnumDescriptor(_ROLE)

Role = enum_type_wrapper.EnumTypeWrapper(_ROLE)
_GENDER = _descriptor.EnumDescriptor(
  name='Gender',
  full_name='cyberbully_detector.Gender',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MALE', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FEMALE', index=1, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1305,
  serialized_end=1335,
)
_sym_db.RegisterEnumDescriptor(_GENDER)

Gender = enum_type_wrapper.EnumTypeWrapper(_GENDER)
_AGE = _descriptor.EnumDescriptor(
  name='Age',
  full_name='cyberbully_detector.Age',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='KID', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEENAGER', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ADULT', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1337,
  serialized_end=1376,
)
_sym_db.RegisterEnumDescriptor(_AGE)

Age = enum_type_wrapper.EnumTypeWrapper(_AGE)
_BULLYINGCLASS = _descriptor.EnumDescriptor(
  name='BullyingClass',
  full_name='cyberbully_detector.BullyingClass',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GOSSIPING', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LAUGHING', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PULLING_HAIR', index=2, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='QUARRELING', index=3, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STABBING', index=4, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ISOLATION', index=5, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PUNCHING', index=6, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SLAPPING', index=7, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STRANGLING', index=8, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NO_BULLYING', index=9, number=10,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1379,
  serialized_end=1547,
)
_sym_db.RegisterEnumDescriptor(_BULLYINGCLASS)

BullyingClass = enum_type_wrapper.EnumTypeWrapper(_BULLYINGCLASS)
_DATACLASS = _descriptor.EnumDescriptor(
  name='DataClass',
  full_name='cyberbully_detector.DataClass',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TRAIN', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEST', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VALIDATION', index=2, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1549,
  serialized_end=1597,
)
_sym_db.RegisterEnumDescriptor(_DATACLASS)

DataClass = enum_type_wrapper.EnumTypeWrapper(_DATACLASS)
PEACE = 1
AFFECTION = 2
ESTEEM = 3
ANTICIPATION = 4
ENGAGEMENT = 5
CONFIDENCE = 6
HAPPINESS = 7
PLEASURE = 8
EXCITEMENT = 9
SURPRISE = 10
SYMPATHY = 11
CONFUSION = 12
DISCONNECTION = 13
FATIGUE = 14
EMBARRASSMENT = 15
YEARNING = 16
DISAPPROVAL = 17
AVERSION = 18
ANNOYANCE = 19
ANGER = 20
SENSITIVITY = 21
SADNESS = 22
DISQUIETMENT = 23
FEAR = 24
PAIN = 25
SUFFERING = 26
DOUBT_CONFUSION = 27
BULLY = 1
VICTIM = 2
BYSTANDER = 3
MALE = 1
FEMALE = 2
KID = 1
TEENAGER = 2
ADULT = 3
GOSSIPING = 1
LAUGHING = 2
PULLING_HAIR = 3
QUARRELING = 4
STABBING = 5
ISOLATION = 6
PUNCHING = 7
SLAPPING = 8
STRANGLING = 9
NO_BULLYING = 10
TRAIN = 1
TEST = 2
VALIDATION = 3



_BBOX = _descriptor.Descriptor(
  name='Bbox',
  full_name='cyberbully_detector.Bbox',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='cyberbully_detector.Bbox.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='cyberbully_detector.Bbox.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='cyberbully_detector.Bbox.width', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='cyberbully_detector.Bbox.height', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=110,
)


_DISCRETEEMOTION = _descriptor.Descriptor(
  name='DiscreteEmotion',
  full_name='cyberbully_detector.DiscreteEmotion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='emotions', full_name='cyberbully_detector.DiscreteEmotion.emotions', index=0,
      number=1, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=112,
  serialized_end=182,
)


_CONTINUOUSEMOTION = _descriptor.Descriptor(
  name='ContinuousEmotion',
  full_name='cyberbully_detector.ContinuousEmotion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='valence', full_name='cyberbully_detector.ContinuousEmotion.valence', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='arousal', full_name='cyberbully_detector.ContinuousEmotion.arousal', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dominance', full_name='cyberbully_detector.ContinuousEmotion.dominance', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=184,
  serialized_end=256,
)


_PERSON = _descriptor.Descriptor(
  name='Person',
  full_name='cyberbully_detector.Person',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='location', full_name='cyberbully_detector.Person.location', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='discrete_emotion', full_name='cyberbully_detector.Person.discrete_emotion', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='continuous_emotion', full_name='cyberbully_detector.Person.continuous_emotion', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='role', full_name='cyberbully_detector.Person.role', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gender', full_name='cyberbully_detector.Person.gender', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='age', full_name='cyberbully_detector.Person.age', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=259,
  serialized_end=569,
)


_ANNOTATION = _descriptor.Descriptor(
  name='Annotation',
  full_name='cyberbully_detector.Annotation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bullying_class', full_name='cyberbully_detector.Annotation.bullying_class', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='people', full_name='cyberbully_detector.Annotation.people', index=1,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='file_path', full_name='cyberbully_detector.Annotation.file_path', index=2,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset', full_name='cyberbully_detector.Annotation.dataset', index=3,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_size', full_name='cyberbully_detector.Annotation.image_size', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_class', full_name='cyberbully_detector.Annotation.data_class', index=5,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fold', full_name='cyberbully_detector.Annotation.fold', index=6,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=572,
  serialized_end=838,
)

_DISCRETEEMOTION.fields_by_name['emotions'].enum_type = _EMOTIONCLASS
_PERSON.fields_by_name['location'].message_type = _BBOX
_PERSON.fields_by_name['discrete_emotion'].message_type = _DISCRETEEMOTION
_PERSON.fields_by_name['continuous_emotion'].message_type = _CONTINUOUSEMOTION
_PERSON.fields_by_name['role'].enum_type = _ROLE
_PERSON.fields_by_name['gender'].enum_type = _GENDER
_PERSON.fields_by_name['age'].enum_type = _AGE
_ANNOTATION.fields_by_name['bullying_class'].enum_type = _BULLYINGCLASS
_ANNOTATION.fields_by_name['people'].message_type = _PERSON
_ANNOTATION.fields_by_name['image_size'].message_type = common__pb2._SIZE
_ANNOTATION.fields_by_name['data_class'].enum_type = _DATACLASS
DESCRIPTOR.message_types_by_name['Bbox'] = _BBOX
DESCRIPTOR.message_types_by_name['DiscreteEmotion'] = _DISCRETEEMOTION
DESCRIPTOR.message_types_by_name['ContinuousEmotion'] = _CONTINUOUSEMOTION
DESCRIPTOR.message_types_by_name['Person'] = _PERSON
DESCRIPTOR.message_types_by_name['Annotation'] = _ANNOTATION
DESCRIPTOR.enum_types_by_name['EmotionClass'] = _EMOTIONCLASS
DESCRIPTOR.enum_types_by_name['Role'] = _ROLE
DESCRIPTOR.enum_types_by_name['Gender'] = _GENDER
DESCRIPTOR.enum_types_by_name['Age'] = _AGE
DESCRIPTOR.enum_types_by_name['BullyingClass'] = _BULLYINGCLASS
DESCRIPTOR.enum_types_by_name['DataClass'] = _DATACLASS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Bbox = _reflection.GeneratedProtocolMessageType('Bbox', (_message.Message,), dict(
  DESCRIPTOR = _BBOX,
  __module__ = 'labels_pb2'
  # @@protoc_insertion_point(class_scope:cyberbully_detector.Bbox)
  ))
_sym_db.RegisterMessage(Bbox)

DiscreteEmotion = _reflection.GeneratedProtocolMessageType('DiscreteEmotion', (_message.Message,), dict(
  DESCRIPTOR = _DISCRETEEMOTION,
  __module__ = 'labels_pb2'
  # @@protoc_insertion_point(class_scope:cyberbully_detector.DiscreteEmotion)
  ))
_sym_db.RegisterMessage(DiscreteEmotion)

ContinuousEmotion = _reflection.GeneratedProtocolMessageType('ContinuousEmotion', (_message.Message,), dict(
  DESCRIPTOR = _CONTINUOUSEMOTION,
  __module__ = 'labels_pb2'
  # @@protoc_insertion_point(class_scope:cyberbully_detector.ContinuousEmotion)
  ))
_sym_db.RegisterMessage(ContinuousEmotion)

Person = _reflection.GeneratedProtocolMessageType('Person', (_message.Message,), dict(
  DESCRIPTOR = _PERSON,
  __module__ = 'labels_pb2'
  # @@protoc_insertion_point(class_scope:cyberbully_detector.Person)
  ))
_sym_db.RegisterMessage(Person)

Annotation = _reflection.GeneratedProtocolMessageType('Annotation', (_message.Message,), dict(
  DESCRIPTOR = _ANNOTATION,
  __module__ = 'labels_pb2'
  # @@protoc_insertion_point(class_scope:cyberbully_detector.Annotation)
  ))
_sym_db.RegisterMessage(Annotation)


# @@protoc_insertion_point(module_scope)
