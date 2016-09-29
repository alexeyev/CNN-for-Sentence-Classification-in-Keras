import numpy as np

import data_helpers
from model import build_compiled_model
from w2v import train_word2vec
import argparse as ap

parser = ap.ArgumentParser(description='CNN for sentence classtion')

parser.add_argument('--epochs', type=int,
                    default=50,
                    help='default=50; epochs count')

parser.add_argument('--dataset', type=str,
                    choices=['okstatus', 'okuser'],
                    default='okstatus',
                    help='default=okstatus, choose dataset')

parser.add_argument('--maxlen', type=int,
                    default=56,
                    help='default=56; max sentence length')

parser.add_argument('--optimizer', type=str, choices=['adam', 'adagrad', 'rmsprop', 'adadelta'],
                    default='adadelta',
                    help='default=adadelta; keras optimizer')

parser.add_argument('--batch', type=int,
                    default=50, # same as in the paper
                    help='default=50; training batch size')

parser.add_argument('--embedding_dim', type=int,
                    default=300, # same as in the paper
                    help='default=50; embedding size')

parser.add_argument('--num_filters', type=int,
                    default=100, # same as in the paper
                    help='default=100; filters number')

parser.add_argument('--gpu_fraction', type=float,
                    default=0.2,
                    help='default=0.2; GPU fraction, please, use with care')

parser.add_argument('--variation', type=str, choices=['CNN-static', 'CNN-rand', 'CNN-non-static'],
                    default='CNN-static', help='default=CNN-static')

args = parser.parse_args()

# ------- setting read args ---------------------

np.random.seed(2)

model_variation = args.variation

# model hyperparameters
# todo: should have l2 norm constraints for regularization (paper, section 2.1)
sequence_length = int(args.maxlen)
embedding_dim = int(args.embedding_dim)  # 20

filter_sizes = (3, 4, 5)  # (3, 4)
num_filters = int(args.num_filters) # 150
dropout_prob = (0.5, 0.5, 0.5)
hidden_dims = 150

# training parameters
batch_size = int(args.batch)
num_epochs = int(args.epochs)
val_split = 0.1

# w2v parameters, see train_word2vec
# todo: probably should not do word2vec stuff here in main.py?
min_word_count = 1  # Minimum word count                        
context = 10  # Context window size

# Load data
print("Loading data...")

# reading texts and labels
(xt, yt), (x_test, y_test) = data_helpers.load_ok_data_gender()

xt, yt, x_test, y_test, vocabulary, vocabulary_inv = data_helpers.build_word_level_data((xt, yt), (x_test, y_test))

if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
    embedding_weights = train_word2vec(xt, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation == 'CNN-static':
        xt = embedding_weights[0][xt]
elif model_variation == 'CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')

# todo: Shuffle data
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

model = build_compiled_model(model_variation, sequence_length, embedding_dim,
                             filter_sizes, num_filters, vocabulary,
                             embedding_weights, dropout_prob, hidden_dims)

# Training model
model.fit(xt, yt,
          batch_size=batch_size,
          nb_epoch=num_epochs,
          validation_split=val_split,
          verbose=2)

res = model.test_on_batch(x_test, y_test)

print(res)
