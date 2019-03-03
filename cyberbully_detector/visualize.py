from .labels_pb2 import Annotation
from PIL import Image, ImageDraw, ImageColor
from pathlib import Path
import numpy as np

def draw_boxes_on_image(img, annotation):
  draw = ImageDraw.Draw(img)

  for person in annotation.people:
    if person.HasField("location"):
      pt_1 = (person.location.x, person.location.y)
      pt_2 = (person.location.x + person.location.width,
              person.location.y + person.location.height)
      draw.rectangle([pt_1, pt_2], outline=ImageColor.getrgb("red"))
  return img

def load_annotated_img(annotation, data_path="data"):
  data_path = Path(data_path)
  img_path = data_path.joinpath(annotation.file_path)
  assert img_path.is_file()

  img = Image.open(str(img_path)).convert("RGB")
  return draw_boxes_on_image(img, annotation)

def np_array_to_img(arr):
  assert len(arr.shape) == 3 # X Y COLOR
  # width first, channels last
  assert arr.shape[-1] == 3  # RGB
  arr *= 255
  return Image.fromarray(arr.astype(np.int8), "RGB")

def image_to_np_array(img):
  arr = np.array(img, dtype=np.float32)
  # put channels first, width last
  arr /= 255
  return arr
