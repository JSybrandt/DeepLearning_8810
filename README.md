
# Cyberbullying Detector


Hey! Here's Justin Sybrandt's midterm submission for DL 8810.

I'll cut right to the case and explain how to run this thing, then I'll go over
how I built this model, and follow up with some references.

## Running this thing

Firstly, you'll need to create a directory `models` that should contain the
pre-trained models uploaded to google drive. Inside this directory should be a
number of sub-folders. My prediction is going to leverage all of these
sub-models, so make sure its all in the right place.

The correct file hierarchy is as follows:

```
DeepLearning_8810/
  predict
  cyberbully_detector/
    ...
  configs/
    ...
  models/
    fold_0/
      ...
    fold_1/
      ...
    fold_2/
      ...
```

Then, make sure the dependencies are setup. This should be as easy as :
```pip install -U -r requirements.txt```

Note that I built these models using tensorflow 1.12.0. The other listed
dependencies are mostly used to coordinate training, however they will be
necessary to launch the module. `protoc` is only needed to build the protocol
buffers needed for training.

Finally, calling *`./predict <img_path>`* should be all you need to do!

## Model summary

I trained a 3-fold cross-validation model. Each fold implements transfer
learning from VGG19 followed by two dense layers and dropout before a simple
10-class softmax. When training, I simply rescale each image to the desired size
(224 x 224) and apply random flips. However, I enforce that each training batch
contains balanced classes. This allows my model to overcome the imbalanced class
problem between bullying and non-bullying images.

When making a prediction, the dropout layer is disabled and all folds process
all flips of the input image. Within a fold, I create a normalized prediction
for the distorted input images. From there each fold's prediction is joined to
make an overall prediction.

I leverage the Stanford 40-class human action dataset to fill in my non-bullying
pictures. I selected this because it was significantly large and contains
non-bullying actions, such as cooking, that are similar to bullying images, such
as stabbing (both contain knives). This is supposed to require that my model
learns more sophisticated relationships that "knives are bad."

## Whats up with this package?

In order to quickly iterate different model architectures. I implemented my own
high-level model syntax powered by protocol buffers! My model is defined in
`configs/midterm.conf` which is then interpreted by `train.py` to produce a
tensorflow computation graph. This way I can select a transfer model, define
layers, and handle multiple convolutions all though an easy-to-edit syntax.
Furthermore I can store these model specifications with their performance
statistics in `mongodb`, which allows me to track model performance over many
many many iterations.

I followed a similar pattern when defining my data. `labels.proto` describes all
relevant information related to an annotation. This is current built to
eventually facilitate bounding boxes, and human emotions (coming soon in the
final project). Again, because I can store protocol buffers in `mongodb` I keep
a central annotation database each worker can access. For example,
`generator.py` looks up the annotation ids from this database in order to
coordinate between datasets, folds, and other training information. In
`proto_util.py` I wrote some conversion code that can transform one of these
annotation objects into a numpy label and back, although I'm not using all of
this code currently. Also, the generator class is also thread-safe, which allows
me to run parallel batch generation in `train.py`.


# Technical Sources

https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html

https://github.com/tensorflow/models/tree/master/research/slim#Pretrained

https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4

https://github.com/taehoonlee/tensornets

http://ruder.io/optimizing-gradient-descent/index.html#adagrad

https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


