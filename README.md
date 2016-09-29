# Convolutional Neural Networks for Sentence Classification

Training convolutional network for classification tasks. 


Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim, [link](http://arxiv.org/pdf/1408.5882v2.pdf). Inspired by Denny Britz article "Implementing a CNN for Text Classification in TensorFlow", [link](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). 


Code is a fork of this implementation [link](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)

## What's different in this fork

* Python3
* a focus on reusing pretrained word vectors
* a few code-decomposition-related modifications
* using argparse for remote configurable execution
* trying to implement configuration as close to the one in the paper as possible (comments are welcome)
* methods for working with datasets in a standard .csv (text, class_label) format

## TODO

* L2 regularization
* ...

## Dependencies

* [Keras](http://keras.io/) 
* [Theano](http://deeplearning.net/software/theano/install.html#install) / [Tensorflow](https://www.tensorflow.org/)