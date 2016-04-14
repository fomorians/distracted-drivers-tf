import os
import time
import numpy as np
import tensorflow as tf

from layers import Input
from utilities import batch_iterator, mkdirp

from sklearn.metrics import log_loss
from keras.preprocessing.image import ImageDataGenerator

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')

DOWNSAMPLE = 20
WIDTH, HEIGHT, NUM_CHANNELS = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE, 3
NUM_CLASSES = 10

LEARNING_RATE = 1e-3

class Model:
    def __init__(self, layers, fold_index, batch_size):
        self.sess = tf.get_default_session()
        self.batch_size = batch_size
        self.fold_index = fold_index

        checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}'.format(self.fold_index))
        mkdirp(checkpoint_path)
        self.checkpoint_dest = os.path.join(checkpoint_path, 'checkpoint')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        global_step_op = self.global_step.assign_add(1)

        self.is_training = tf.placeholder(tf.bool, shape=[])

        self.x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, NUM_CHANNELS])
        self.y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        self.layers = [Input(self.x)] + layers

        prev_y = None
        for i, layer in enumerate(self.layers):
            prev_y = layer.apply(prev_y, i, self)
        self.y = prev_y

        with tf.name_scope("loss"):
            self.loss_op = -tf.reduce_sum(self.y_ * tf.log(self.y + 1e-12))
            tf.scalar_summary("loss", self.loss_op)

            loss_ema = tf.train.ExponentialMovingAverage(decay=0.9, num_updates=self.global_step)
            loss_ema_op = loss_ema.apply([self.loss_op])
            tf.scalar_summary('loss_ema', loss_ema.average(self.loss_op))

        with tf.name_scope("test"):
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))

            self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.scalar_summary('accuracy', self.accuracy_op)

            accuracy_ema = tf.train.ExponentialMovingAverage(decay=0.9, num_updates=self.global_step)
            accuracy_ema_op = accuracy_ema.apply([self.accuracy_op])
            tf.scalar_summary('accuracy_ema', accuracy_ema.average(self.accuracy_op))

        with tf.control_dependencies([global_step_op, accuracy_ema_op, loss_ema_op]):
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss_op, name='train')

        self.summaries_op = tf.merge_all_summaries()

        self.saver = tf.train.Saver(max_to_keep=1)

        self.sess.run(tf.initialize_all_variables())

        summary_run_path = os.path.join(SUMMARY_PATH, str(int(time.time())))
        self.summary_writer = tf.train.SummaryWriter(summary_run_path, self.sess.graph_def)

        tf.train.write_graph(self.sess.graph_def, MODEL_PATH, 'model.pb', as_text=False)

        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dest)
        print('Attempting to restore {}...'.format(latest_checkpoint_path))
        if latest_checkpoint_path:
            print('Restoring checkpoint: {}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)
        else:
            print('Could not find checkpoint to restore.')

    def train(self, X_train, y_train, epoch):
        batch_index = 0

        summary_interval = 10

        mean = X_train.mean()
        std = X_train.std()

        for batch_x, batch_y in batch_iterator(X_train, y_train, batch_size=self.batch_size, shuffle_batch=True):
            batch_x = batch_x.astype(np.float32)
            batch_x = (batch_x - mean) / std

            start_time = time.time()

            _, loss, accuracy, summary, global_step = self.sess.run([
                self.train_op,
                self.loss_op,
                self.accuracy_op,
                self.summaries_op,
                self.global_step
            ], feed_dict={
                self.x: batch_x,
                self.y_: batch_y,
                self.is_training: True
            })

            elapsed_time = time.time() - start_time

            print('Training: fold: {}, epoch: {}, global step: {}, loss: {}, accuracy: {} (duration: {})'.format(self.fold_index, epoch, global_step, loss, accuracy, elapsed_time))

            if batch_index % summary_interval == 0:
                self.summary_writer.add_summary(summary, global_step=global_step)

            batch_index += 1

        self.saver.save(self.sess, self.checkpoint_dest, global_step=global_step)

    def validate(self, X_valid, y_valid):
        num_samples = X_valid.shape[0]

        predictions = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)
        accuracy = 0
        loss = 0

        mean = X_valid.mean()
        std = X_valid.std()

        num_batches = 0
        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)

            batch_x = X_valid[batch_start:batch_end, ...]
            batch_x = batch_x.astype(np.float32)
            batch_x = (batch_x - mean) / std

            batch_y = y_valid[batch_start:batch_end, ...]

            batch_predictions, batch_loss, batch_accuracy = self.sess.run([
                self.y,
                self.loss_op,
                self.accuracy_op
            ], feed_dict={
                self.x: batch_x,
                self.y_: batch_y,
                self.is_training: False
            })

            predictions[batch_start:batch_end] = batch_predictions
            accuracy += batch_accuracy
            loss += batch_loss

            num_batches += 1

        score = log_loss(y_valid, predictions)
        accuracy /= num_batches

        return loss, accuracy, score

    def evaluate(self, X_test):
        num_samples = X_test.shape[0]

        predictions = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)
        num_batches = num_samples // self.batch_size

        mean = X_test.mean()
        std = X_test.std()

        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)

            batch_x = X_test[batch_start:batch_end, ...]
            batch_x = batch_x.astype(np.float32)
            batch_x = (batch_x - mean) / std

            predictions[batch_start:batch_end, :] = self.sess.run(self.y, feed_dict={
                self.x: batch_x,
                self.is_training: False
            })

        return predictions
