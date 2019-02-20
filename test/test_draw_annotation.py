#!/usr/bin/env python3

from cyberbully_detector.data_util import load_annotation
from cyberbully_detector.data_util import get_annotation_ids
from cyberbully_detector.visualize import load_annotated_img

ids = get_annotation_ids("TRAIN", "emotic")
annotation = load_annotation(ids[0])
img = load_annotated_img(annotation, "../data")
img.show()

ids = get_annotation_ids("TRAIN", "stanford_40")
annotation = load_annotation(ids[0])
img = load_annotated_img(annotation, "../data")
img.show()

