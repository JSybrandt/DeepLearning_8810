from .bully_pb2 import GRAYSCALE, RGBA, RGB, RBG
from .bully_pb2 import Config
from .proto_util import PROTO_PARSERS
from .proto_util import get_or_none
from .labels_pb2 import Annotation
from .labels_pb2 import DataClass
import logging as log
import multiprocessing
from pymongo import MongoClient
from google.protobuf.json_format import Parse as json2proto
import json
import logging as log


def get_annotation_db_connection(host="jcloud", db_name="DL_8810"):
  return MongoClient(host)[db_name]

def get_annotation_ids(db, data_class=None, datasets=None, folds=None):
  "If datasets is a string, we will only return annotations from that set "
  "matching the class"
  if type(data_class) == int:
    data_class == DataClass.Name(data_class)

  query = {"$and":[]}
  if data_class is not None:
    query["dataClass"] = data_class

  if datasets is not None:
    if type(datasets) == str:
      query["dataset"] = datasets
    else:
      query["$and"].append({"$or": [{"dataset": ds} for ds in datasets]})

  if folds is not None:
    query["$and"].append({"$or": [{"fold": f} for f in folds]})

  log.info(query)

  res = []
  for query_res in db.annotations.find(query, {"_id":1}):
    res.append(query_res["_id"])
  return res

def load_annotation(db, mongo_id):
  annotation_json = db.annotations.find_one({"_id":mongo_id}, {"_id":0})
  return json2proto(json.dumps(annotation_json), Annotation())

def colormode_to_str(color_mode):
  modes = {}
  modes[GRAYSCALE] = "grayscale"
  modes[RGB] = "rgb"
  modes[RBG] = "rbg"
  modes[RGBA] = "rgba"
  assert color_mode in modes
  return modes[color_mode]

def colormode_to_dim(color_mode):
  modes = {}
  modes[GRAYSCALE] = 1
  modes[RGB] = 3
  modes[RBG] = 3
  modes[RGBA] = 4
  assert color_mode in modes
  return modes[color_mode]

def get_config(args):
  # Default Config
  config = Config()
  if args.config is not None:
    log.info("Parsing config")
    PROTO_PARSERS[args.config.suffix](args.config, config)
  else:
    log.info("Using default config")
  return config

def get_worker_count(config):
  total_cpus = multiprocessing.cpu_count()
  desired_cpu = config.system.workers
  if desired_cpu < 0:
    desired_cpu += total_cpus
  if desired_cpu < 0:
    raise ValueError("Config is asking for fewer workers than available")
  if desired_cpu > total_cpus:
    raise ValueError("Config is asking for more workers than available")
  log.info("Using %s/%s workers", desired_cpu, total_cpus)
  return desired_cpu

def class_name_map(generator):
  return {i: c for c, i in generator.class_indices.items()}
