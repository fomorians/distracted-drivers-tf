from __future__ import print_function
from __future__ import division

import math
import random

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def calc_geom(scores, num_predictions):
    result = scores[0]
    for i in range(1, num_predictions):
        result *= scores[i]
    result = math.pow(result, 1.0 / num_predictions)
    return result

def calc_geom_arr(predictions_total, num_predictions):
    results = np.array(predictions_total[0])
    for i in range(1, num_predictions):
        results *= np.array(predictions_total[i])
    results = np.power(results, 1.0 / num_predictions)
    return results.tolist()

def weight_bias(shape, stddev, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=shape[-1:]), name='bias')
    return W, b

def batch_iterator(X, y, batch_size=None, shuffle_batch=False):
    length = len(X)

    if batch_size is None:
        batch_size = length

    if shuffle_batch:
        X, y = shuffle(X, y)

    for batch_start in range(0, length, batch_size):
        batch_end = batch_start + batch_size
        if batch_end > length:
            continue
        yield X[batch_start:batch_end], y[batch_start:batch_end]

def write_submission(predictions, ids, dest):
    df = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', pd.Series(ids, index=df.index))
    df.to_csv(dest, index=False)
