# coding:utf-8
"""
    Custom datasets reading and preprocessing routines
"""

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd
import pymystem3

mystem = pymystem3.Mystem()


def clean_str(string):
    """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-zА-Яа-я0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels_pos_neg():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, maxlen=56, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length.
    Returns padded sentences.
    """
    sequence_length = maxlen  # max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max(0, sequence_length - len(sentence))
        new_sentence = sentence[:sequence_length] + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_x(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = build_input_x(sentences, vocabulary)
    y = np.array(labels)
    return [x, y]


def load_data_pos_neg():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_pos_neg()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def build_word_level_data(train_data, test_data):
    sentences_train, labels_train = train_data
    sentences_test, labels_test = test_data

    sentences_train = [clean_str(sent) for sent in sentences_train]
    sentences_train = [mystem.lemmatize(s) for s in sentences_train]

    sentences_test = [clean_str(sent) for sent in sentences_test]
    sentences_test = [mystem.lemmatize(s) for s in sentences_test]

    sentences_train_padded = pad_sentences(list(sentences_train))
    sentences_test_padded = pad_sentences(list(sentences_test))

    print(" ".join(sentences_train_padded[0]))

    vocabulary, vocabulary_inv = \
        build_vocab(sentences_train_padded + sentences_test_padded)

    x_train, y_train = build_input_data(sentences_train_padded, labels_train, vocabulary)
    x_test, y_test = build_input_data(sentences_test_padded, labels_test, vocabulary)

    return x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv


def encode_word_level_data(prepared_x, vocabulary):
    x = build_input_x(pad_sentences(list(prepared_x.ix[:, 0])), vocabulary)
    return x


def batch_iter(data, batch_size, num_epochs):
    """
        Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data.shape[0])
    num_batches_per_epoch = int(len(data) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def read_data_file(fname, target_index=0, normalize=True, binary=False):
    content = pd.read_csv(fname, header=None, index_col=False)
    content.dropna(inplace=True)
    content.reset_index(inplace=True, drop=True)

    x = content.ix[:, content.shape[1] - 1]
    x = np.array(x)

    y = content.ix[:, target_index].values + 0.0

    if normalize:
        max_y = np.max(np.abs(y))
        y /= max_y
    if binary:
        vals = list(set(y))
        if len(vals) > 2:
            raise Exception("Binary input data is not binary! Dataset %s, target_index=%d" % (fname, target_index))
        y = np.array([0 if a == vals[0] else 1 for a in y])

    return x, y


def load_ok_data_gender():
    train_data = read_data_file('./data/ok/ok_train.csv', target_index=2, binary=True)
    test_data = read_data_file('./data/ok/ok_test.csv', target_index=2, binary=True)
    return train_data, test_data


def load_ok_user_data_gender():
    train_data = read_data_file('./data/ok/ok_user_train.csv', target_index=2, binary=True)
    test_data = read_data_file('./data/ok/ok_user_test.csv', target_index=2, binary=True)

    return train_data, test_data


def load_sentirueval_data():
    train_data = read_data_file('./data/sentirueval/train.csv')
    test_data = read_data_file('./data/sentirueval/test.csv')
    return train_data, test_data


def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, np.asmatrix(y).T))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi


def clean_data_np(x):
    # load data
    all = [s.strip() for s in list(x)]

    # split by words
    x_text = [clean_str(sent) for sent in all]
    x_text = [s.split(u" ") for s in x_text]
    return x_text


def clean_data_lists(x):
    # load data
    all = [s.strip() for s in x]
    # split by words
    x_text = [clean_str(sent) for sent in all]
    x_text = [s.split(u" ") for s in x_text]
    return x_text


if __name__ == '__main__':
    # read_w2v()
    df = pd.DataFrame([{"x": u"привет"}, {"x": u"пока"}])
