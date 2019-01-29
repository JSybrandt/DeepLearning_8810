from google.protobuf import json_format
import json

def parse_pb_to_proto(path, proto_obj):
  assert path.suffix == ".pb"
  with open(path, 'rb') as proto_file:
    proto_obj.ParseFromString(proto_file.read())

def parse_json_to_proto(path, proto_obj):
  assert path.suffix == ".json"
  with open(path, 'r') as json_file:
    json_format.Parse(json_file.read(), proto_obj)

def get_or_none(proto, field_name):
  if proto.HasField(field_name):
    return getattr(proto, field_name)
  else:
    return None

PROTO_PARSERS = {
    ".pb": parse_pb_to_proto,
    ".json": parse_json_to_proto,
}
