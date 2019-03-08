#!/bin/bash

ssh jcloud 'mongo DL_8810 --eval "db.annotations.deleteMany({})"'

pushd emotic
./convert_annotations.py
popd

pushd stanford_40
./convert_annotations.py
popd

pushd bullying
./convert_annotations.py
popd
