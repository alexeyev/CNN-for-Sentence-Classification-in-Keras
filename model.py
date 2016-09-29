# coding: utf-8
"""
    todo: should have l2 norm constraints for regularization (paper, section 2.1)
"""

from keras.layers import \
    Activation, Dense, Dropout, \
    Embedding, Input, Merge, Convolution1D
from keras.layers.core import Lambda
from keras.models import Sequential, Model
from keras import backend as K


def max_1d(X):
    """
        Max-over-time pooling
    """
    return K.max(X, axis=1)


def build_compiled_model(model_variation, sequence_length, embedding_dim,
                         filter_sizes, num_filters, vocabulary, embedding_weights,
                         dropout_prob, hidden_dims, optimizer):
    # graph subnet with one input and one output,
    # convolutional layers concatenated in parallel
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []

    # we build a convolution for each filter size
    for filter_size in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=filter_size,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
        # could be like that
        # pool = MaxPooling1D(pool_length=2)(conv)
        # but the paper suggested max-over-time pooling
        # see https://github.com/fchollet/keras/commit/85f80714c29d6bd8c8cde9138f0369f5df1b9a33
        pool = Lambda(max_1d, output_shape=(num_filters,))(conv)
        # todo: check if everything is ok with dimensions
        # flatten = Flatten()(pool)
        convs.append(pool)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()

    if model_variation != 'CNN-static':
        model.add(Embedding(len(vocabulary), embedding_dim,
                            input_length=sequence_length,
                            weights=embedding_weights))

    # NB! the paper says there's dropout on penultimate level

    # not sure whether this was in the original paper
    # model.add(Dropout(p=dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(output_dim=hidden_dims))
    model.add(Dropout(p=dropout_prob[1]))
    model.add(Activation(activation='relu'))
    model.add(Dense(output_dim=1))
    # not sure whether this was in the original paper
    # model.add(Dropout(p=dropout_prob[2]))
    model.add(Activation(activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model