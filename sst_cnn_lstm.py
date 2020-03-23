from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging

import pickle
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Conv1D, MaxPooling1D
from keras.preprocessing import sequence

from Attention_layer import AttentionM
from utils import evaluate

hidden_dim = 120
kernel_size = 3
nb_filter = 120

batch_size = 32
epochs = 10

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

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_test, y_dev = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev[option]

        if rev['split'] == 'train':
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 'dev':
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == 'test':
            X_test.append(sent)
            y_test.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_dev = np.array(y_dev)

    return [X_train, X_test, X_dev, y_train, y_test, y_dev]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'sst_glove.pickle3')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    # datasets
    X_train, X_test, X_dev, y_train, y_test, y_dev = make_idx_data(
        revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info(
        "dimension num of word vector [num_features]: %d" % num_features)

    # Keras Model
    sequence = Input(shape=(maxlen, ), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features,
                         input_length=maxlen, weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    convolution = Conv1D(filters=nb_filter,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1
                            ) (embedded)

    maxpooling = MaxPooling1D(pool_size=2) (convolution)
    # maxpooling = Flatten() (maxpooling)
    # maxpooling = Reshape((int(maxpooling.shape[2]), int(maxpooling.shape[1]))) (maxpooling)

    lstm = LSTM(hidden_dim, recurrent_dropout=0.25) (maxpooling)

    output = Dense(1, activation='linear')(lstm)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=[X_dev, y_dev], verbose=1)

    y_pred = model.predict(X_test, batch_size=batch_size).flatten()
    evaluate(y_test, y_pred)

    output_file = os.path.join('result', 'sst_lstm_' + option + '.pickle3')
    pickle.dump([y_pred], open(output_file, 'wb'))
    logging.info('result output!')
