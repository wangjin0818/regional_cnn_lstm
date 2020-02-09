from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import gensim
import pickle
import re

import numpy as np
import pandas as pd

from collections import defaultdict
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


import nltk
from nltk.tokenize import WordPunctTokenizer
sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# configuration
num_feature = 300


def rescale(value):
    return value * 8. + 1.


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    # string = re.sub(r"[^A-Za-z0-9]", " ", string)
    # words = string.strip().split()
    # new_words = []
    # for word in words:
    #     if word in exception_word:
    #         word = word.replace('-', ' ').replace('\/', ' ')

    #     new_words.append(word)

    # string = ' '.join(new_words)

    # string = string.replace('-', ' ').replace('\/', ' ')
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)

    return string.strip().lower()


def build_data_train_test(x_train, y_train_valence, \
        x_test, y_test_valence,                     \
        x_valid, y_valid_valence):

    revs = []
    vocab = defaultdict(float)
    max_len = []

    for i in range(len(x_train)):
        orig_rev = clean_str(x_train[i])

        region_rev = sen_tokenizer.tokenize(orig_rev)
        for region in region_rev:
            max_len.append(len(region.split()))

        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {
            'valence': rescale(y_train_valence[i]),
            'region_text': region_rev,
            'num_regions': len(region_rev),
            'num_words': len(orig_rev.split()),
            'split': 'train'
        }
        revs.append(datum)

    for i in range(len(x_valid)):
        orig_rev = clean_str(x_valid[i])

        region_rev = sen_tokenizer.tokenize(orig_rev)
        for region in region_rev:
            max_len.append(len(region.split()))

        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {
            'valence': rescale(y_valid_valence[i]),
            'region_text': region_rev,
            'num_regions': len(region_rev),
            'num_words': len(orig_rev.split()),
            'split': 'valid'
        }
        revs.append(datum)


    for i in range(len(x_test)):
        orig_rev = clean_str(x_test[i])

        region_rev = sen_tokenizer.tokenize(orig_rev)
        for region in region_rev:
            max_len.append(len(region.split()))

        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {
            'valence': rescale(y_test_valence[i]),
            'region_text': region_rev,
            'num_regions': len(region_rev),
            'num_words': len(orig_rev.split()),
            'split': 'test'
        }
        revs.append(datum)

    return revs, vocab, np.max(max_len)

def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1
    
    logging.info('unk words: %d' % (unk_words))
    return word_vecs

def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size+2, k), dtype=np.float32)
    W[0] = np.zeros((k, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    sst_file = os.path.join('pickle', 'SST.pickle3')
    (x_train, y_train_valence, y_train_labels,
        x_test, y_test_valence, y_test_labels,
        x_valid, y_valid_valence, y_valid_labels,
        x_train_polarity, y_train_polarity,
        x_test_polarity, y_test_polarity,
        x_valid_polarity, y_valid_polarity) = pickle.load(open(sst_file, 'rb'))

    revs, vocab, max_l = build_data_train_test(x_train, y_train_valence, \
                                        x_test, y_test_valence,  \
                                        x_valid, y_valid_valence)

    max_r = np.max(pd.DataFrame(revs)['num_regions'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))
    logging.info('max region length: ' + str(max_r))

    # word2vec GoogleNews
    # model_file = os.path.join('vector', 'GoogleNews-vectors-negative300.bin')
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    # Glove Common Crawl
    model_file = os.path.join('e:\\', 'lib', 'glove.840B.300d.gensim.txt')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    w2v = load_bin_vec(model, vocab)
    logging.info('word embeddings loaded!')
    logging.info('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=model.vector_size)
    logging.info('extracted index from embeddings! ')

    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = os.path.join('pickle', 'sst_region_glove.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l, max_r], open(pickle_file, 'wb'))
    logging.info('dataset created!')