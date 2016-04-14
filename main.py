from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import numpy as np
import tensorflow as tf

from sklearn.cross_validation import LabelShuffleSplit

from model import Model
from utilities import write_submission, calc_geom, calc_geom_arr, mkdirp
from architectures import vgg_bn

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test locally.')

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/data_20.pkl' if not FLAGS.test else 'dataset/data_20_subset.pkl')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')

NUM_EPOCHS = 20 if not FLAGS.test else 1
MAX_FOLDS = 8
BATCH_SIZE = 50

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)

_, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)

predictions_total = []
scores_total = []
num_folds = 0

for train_index, valid_index in LabelShuffleSplit(driver_indices, n_iter=MAX_FOLDS, test_size=0.2, random_state=67):
    print('Running fold...', len(train_index), len(valid_index))

    X_train, y_train = X_train_raw[train_index,...], y_train_raw[train_index,...]
    X_valid, y_valid = X_train_raw[valid_index,...], y_train_raw[valid_index,...]

    with tf.Graph().as_default(), tf.Session() as sess:
        layers = vgg_bn()
        model = Model(layers, num_folds, batch_size=BATCH_SIZE)

        patience = 2
        wait = 0
        best = np.Inf

        print('Begin training...')
        for epoch in range(NUM_EPOCHS):
            model.train(X_train, y_train, epoch)

            print('Begin validation...')
            loss, accuracy, score = model.validate(X_valid, y_valid)

            print('Validation: fold: {}, epoch: {}, loss: {}, accuracy: {}, score: {}'.format(num_folds, epoch, loss, accuracy, score))

            if loss < best:
                print('New best validation loss! Was: {}, Now: {}'.format(best, loss))
                best = loss
                wait = 0
            else:
                wait += 1
                print('Validation loss did not improve for {}/{} epochs.'.format(wait, patience))

            if wait == 2:
                print('Stopping early. Validation loss did not improve for {}/{} epochs.'.format(wait, patience))
                break


        model.summary_writer.close()
        scores_total.append(score)

        print('Begin evaluation...')
        predictions = model.evaluate(X_test)
        predictions_total.append(predictions)

    num_folds += 1

score_geom = calc_geom(scores_total, num_folds)
predictions_geom = calc_geom_arr(predictions_total, num_folds)

print('Writing submission for {} folds, score: {}...'.format(num_folds, score_geom))
submission_dest = os.path.join(SUMMARY_PATH, 'submission_{}_{}.csv'.format(int(time.time()), score_geom))
write_submission(predictions_geom, X_test_ids, submission_dest)

print('Done.')
