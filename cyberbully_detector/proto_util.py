from google.protobuf import json_format
from google.protobuf import text_format
import json

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
