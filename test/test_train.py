from cyberbully_detector.train import index_labels
from cyberbully_detector.bully_pb2 import TrainingConfig

def test_index_labels_typical():
  config = TrainingConfig()
  config.examples.add().label = "A"
  config.examples.add().label = "B"
  config.examples.add().label = "A"
  config.examples.add().label = "B"
  index_labels(config)
  assert len(config.label_map) == 2
  assert "A" in config.label_map
  assert "B" in config.label_map
  assert config.label_map["A"] == 0
  assert config.label_map["B"] == 1
