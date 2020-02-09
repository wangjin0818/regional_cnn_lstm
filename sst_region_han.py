# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, RepeatVector, Permute, TimeDistributed
from keras.preprocessing import sequence

from keras.utils import np_utils
from utils import evaluate
from Attention_layer import AttentionM

batch_size = 128
epochs = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 60

option = 'valence'


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60, max_region=20):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_test, y_dev = [], [], [], [], [], []
    for rev in revs:
        sent = np.zeros((max_region, maxlen))
        region_text = rev['region_text']
        for i, region in enumerate(region_text):
            if i >= max_region:
                continue

            for j, word in enumerate(region.split()):
                if word in word_idx_map:
                    sent[i, j] = word_idx_map[word]
                else:
                    sent[i, j] = 1

        y = rev[option]

        if rev['split'] == 'train':
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 'valid':
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == 'test':
            X_test.append(sent)
            y_test.append(y)

    X_train = np.array(X_train, dtype='int')
    X_dev = np.array(X_dev, dtype='int')
    X_test = np.array(X_test, dtype='int')
    # X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    # X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    # X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_test, y_dev]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'sst_region_glove.pickle3')
    revs, W, word_idx_map, vocab, max_l, max_r = pickle.load(
        open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_test, y_dev = make_idx_data(
        revs, word_idx_map, maxlen=max_l, max_region=max_r)
    print(X_train.shape)
    print(X_dev.shape)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_region = X_train.shape[1]     # 200
    logging.info("len_region [len_region]: %d" % len_region)

    len_sentence = X_train.shape[2]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info(
        "dimension num of word vector [num_features]: %d" % num_features)

    sentence_input = Input(shape=(max_l,), dtype='int32')
    embedded_sequences = Embedding(input_dim=max_features, output_dim=num_features,
                                   input_length=max_l, mask_zero=True, weights=[W], trainable=False)(sentence_input)
    l_lstm = Bidirectional(
        LSTM(hidden_dim, return_sequences=True))(embedded_sequences)
    l_att = AttentionM()(l_lstm)
    l_fc = Dense(hidden_dim * 2, activation="tanh")(l_att)
    sentEncoder = Model(sentence_input, l_fc)

    review_input = Input(shape=(max_r, max_l), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(
        LSTM(hidden_dim, return_sequences=True))(review_encoder)
    l_att_sent = AttentionM()(l_lstm_sent)
    preds = Dense(1, activation='linear')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    print("Hierachical Attention LSTM")
    model.summary()

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=[X_dev, y_dev], verbose=1)
    y_pred = model.predict(X_test, batch_size=batch_size).flatten()
    evaluate(y_test, y_pred)

    output_file = os.path.join('result', 'sst_han_' + option + '.pickle3')
    pickle.dump([y_pred], open(output_file, 'wb'))
    logging.info('result output!')
