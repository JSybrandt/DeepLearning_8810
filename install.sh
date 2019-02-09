#!/bin/bash

echo "Checking for protoc"
which protoc
if [[ $? == 1 ]]; then
  echo "Failed to detect protoc : the proto compiler"
  echo "sudo apt install protoc"
  exit 1
fi

protoc --python_out cyberbully_detector bully.proto

pip install --user -r requirements.txt
pip install --user -e .

