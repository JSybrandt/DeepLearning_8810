#!/bin/bash

MODEL=$(mktemp -u --suffix=.h5)

python3 -m cyberbully_detector \
           $MODEL \
           -v \
           -d ../data/ \
           --config ../configs/tiny_test_config.txt \
           train

python3 -m cyberbully_detector \
           $MODEL \
           -v \
           -d ../data/ \
           --config ../configs/tiny_test_config.txt \
           test
