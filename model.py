from keras.layers import \
    Activation, Dense, Dropout, \
    Embedding, Flatten, Input, \
    Merge, Convolution1D, MaxPooling1D
from keras.models import Sequential, Model


def build_compiled_model(model_variation, sequence_length, embedding_dim,
                         filter_sizes, num_filters, vocabulary, embedding_weights,
                         dropout_prob, hidden_dims):

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
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

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

    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(output_dim=hidden_dims))
    model.add(Dropout(p=dropout_prob[1]))
    model.add(Activation(activation='relu'))
    model.add(Dense(output_dim=1))
    model.add(Activation(activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model
