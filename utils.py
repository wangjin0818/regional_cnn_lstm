from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import logging

import pickle
import numpy as np

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr, kendalltau


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    pr = pearsonr(y_true, y_pred)[0]
    sr = spearmanr(y_true, y_pred)[0]
    kt = kendalltau(y_true, y_pred)[0]

    print('Evaluation length: %d' % len(y_pred))
    print('MAE: %.3f' % (mae))
    print('Pearsonr: %.3f' % (pr))
    print('Kendalltau: %.3f' % (kt))
    print('Spearmanr: %.3f' % (sr))


def evaluate_multi(y_true, y_pred):
    y_val_pred = []
    y_val_true = []
    y_aro_pred = []
    y_aro_true = []

    # print(y_true.shape)
    print(y_pred.shape)

    for i in range(len(y_true)):
        y_val_pred.append(y_pred[i][0])
        y_aro_pred.append(y_pred[i][1])
        y_val_true.append(y_true[i][0])
        y_aro_true.append(y_true[i][1])

    mae = mean_absolute_error(y_val_true, y_val_pred)
    pr = pearsonr(y_val_true, y_val_pred)[0]
    sr = spearmanr(y_val_true, y_val_pred)[0]
    kt = kendalltau(y_val_true, y_val_pred)[0]

    logging.info('Evaluation length: %d' % len(y_pred))
    logging.info('valence:')
    logging.info('MAE: %.3f' % (mae))
    logging.info('Pearsonr: %.3f' % (pr))
    logging.info('Kendalltau: %.3f' % (kt))
    logging.info('Spearmanr: %.3f' % (sr))

    mae = mean_absolute_error(y_aro_true, y_aro_pred)
    pr = pearsonr(y_aro_true, y_aro_pred)[0]
    sr = spearmanr(y_aro_true, y_aro_pred)[0]
    kt = kendalltau(y_aro_true, y_aro_pred)[0]

    logging.info('Evaluation length: %d' % len(y_pred))
    logging.info('arousal:')
    logging.info('MAE: %.3f' % (mae))
    logging.info('Pearsonr: %.3f' % (pr))
    logging.info('Kendalltau: %.3f' % (kt))
    logging.info('Spearmanr: %.3f' % (sr))


def removeRepeatedElem(string):
    strList = list(string)
    tempStr = []

    [tempStr.append(i) for i in strList if not i in tempStr]
    return ''.join(tempStr)


def output_result(file_name, y_pred):
    with open(file_name, 'w') as my_file:
        for i, value in enumerate(y_pred):
            my_file.write('%.3f\n' % (value))
