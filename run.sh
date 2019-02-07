#!/bin/bash

protoc --python_out cyberbully_detector/ bully.proto

python3 -m cyberbully_detector $@
