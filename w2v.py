import os
from os.path import join, exists, split

import numpy as np
from gensim.models import word2vec


def read_custom_w2v_noparams(filepath):
    model = word2vec.Word2Vec.load_word2vec_format(filepath, binary=True)
    return model


def train_word2vec(sentence_matrix, vocabulary_inv, num_features=300,
                   min_word_count=1, context=10, model_dir='pretrained/word2vec_models'):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """

    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)

    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # set values for various parameters
        num_workers = 3  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # initialize and train the model
        print("Training Word2Vec model...")
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                                            size=num_features, min_count=min_word_count, \
                                            window=context, sample=downsampling)

        # if we don't plan to train the model any further, calling
        # init_sims will make the model more memory-efficient.
        embedding_model.init_sims(replace=True)

        # saving the model for later use
        # you can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)

        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add words trained w2v model doesn't know
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model \
                                       else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) \
                                   for w in vocabulary_inv])]
    return embedding_weights


if __name__ == '__main__':
    import data_helpers

    print("Loading data...")
    x, _, _, vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)
