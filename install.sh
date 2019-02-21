#!/bin/bash

echo "Checking for protoc"
which protoc
if [[ $? == 1 ]]; then
  echo "Failed to detect protoc : the proto compiler"
  echo "sudo apt install protoc"
  exit 1
fi

protoc \
  --python_out cyberbully_detector \
  -I protos \
  protos/bully.proto \
  protos/labels.proto \
  protos/common.proto

# Protoc issue https://github.com/protocolbuffers/protobuf/issues/1491
sed -i -E 's/^import.*_pb2/from . \0/' cyberbully_detector/*_pb2.py

pip install --user -r requirements.txt
pip install --user -e .

