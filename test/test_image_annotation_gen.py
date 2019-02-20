#!/usr/bin/env python3

from cyberbully_detector.data_util import load_annotation
from cyberbully_detector.data_util import get_annotation_ids
from cyberbully_detector.generator import ImageAndAnnotationGenerator
from cyberbully_detector.visualize import draw_boxes_on_image

gen = ImageAndAnnotationGenerator("../data")
images, annotations = next(gen.flow_from_mongo(
    "TRAIN",
    sample_size=(244,244),
    short_side_size=300,
    batch_size=1,
    dataset="emotic"))

img = images[0]
annotation = annotations[0]
img = draw_boxes_on_image(img, annotation)
img.show()

