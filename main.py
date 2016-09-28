import numpy as np

import data_helpers
from model import build_compiled_model
from w2v import train_word2vec

np.random.seed(2)

model_variation = 'CNN-rand'  # CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# model hyperparameters
sequence_length = 56
embedding_dim = 300 # 20
filter_sizes = (3, 4, 5) # (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# training parameters
batch_size = 32
num_epochs = 100
val_split = 0.1

# w2v parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10  # Context window size

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()

if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation == 'CNN-static':
        x = embedding_weights[0][x]
elif model_variation == 'CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

model = build_compiled_model(model_variation, sequence_length, embedding_dim,
                             filter_sizes, num_filters, vocabulary,
                             embedding_weights, dropout_prob, hidden_dims)

# Training model
model.fit(x_shuffled, y_shuffled,
          batch_size=batch_size,
          nb_epoch=num_epochs,
          validation_split=val_split,
          verbose=2)
