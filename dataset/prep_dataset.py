from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('subset', False, 'If true, build a subset.')
flags.DEFINE_integer('downsample', 20, 'Downsample factor.')

WIDTH, HEIGHT = 640 // FLAGS.downsample, 480 // FLAGS.downsample
NUM_CLASSES = 10

def load_image(path):
    return imresize(imread(path), (HEIGHT, WIDTH))

def load_train(base):
    driver_imgs_list = pd.read_csv('driver_imgs_list.csv')
    driver_imgs_grouped = driver_imgs_list.groupby('classname')

    X_train = []
    y_train = []
    driver_ids = []

    print('Reading train images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        paths = glob.glob('{}c{}/*.jpg'.format(base, j))
        driver_ids_group = driver_imgs_grouped.get_group('c{}'.format(j))

        if FLAGS.subset:
            paths = paths[:100]
            driver_ids_group = driver_ids_group.iloc[:100]

        driver_ids += driver_ids_group['subject'].tolist()

        for i, path in tqdm(enumerate(paths), total=len(paths)):
            img = load_image(path)
            if i == 0:
                imsave('c{}.jpg'.format(j), img)

            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = OneHotEncoder(n_values=NUM_CLASSES) \
        .fit_transform(y_train.reshape(-1, 1)) \
        .toarray()

    return X_train, y_train, driver_ids

def load_test(base):
    X_test = []
    X_test_id = []
    paths = glob.glob('{}*.jpg'.format(base))

    if FLAGS.subset:
        paths = paths[:100]

    print('Reading test images...')
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        id = os.path.basename(path)
        img = load_image(path)

        X_test.append(img)
        X_test_id.append(id)

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    return X_test, X_test_id

X_train, y_train, driver_ids = load_train('imgs/train/')
X_test, X_test_ids = load_test('imgs/test/')

if FLAGS.subset:
    dest = 'data_{}_subset.pkl'.format(FLAGS.downsample)
else:
    dest = 'data_{}.pkl'.format(FLAGS.downsample)

with open(dest, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, X_test_ids, driver_ids), f)
