# encoding=utf-8

import tensorflow as tf
import sys

reload(sys)
sys.setdefaultencoding("utf-8")
from tensorflow.contrib.layers import xavier_initializer


class RNNModel():
    def __init__(self, rnn_size, embedding_size, class_num, vocab_size, learning_rate, model_path):
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.class_num = class_num
        self.global_step = tf.Variable(tf.constant(0), trainable=False)
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.96, staircase=True)
        self.build_model()

    def build_model(self):
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_data")
        self.input_length = tf.placeholder(tf.int32, [None], name="input_length")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.class_num], name="labels")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name="batch_size")

        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32,
                                    initializer=xavier_initializer())
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
        self.cell_state = cell.zero_state(self.batch_size, tf.float32)
        outputs, self.last_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.input_length, dtype=tf.float32)

        last = self._extract_axis_1(outputs, self.input_length - 1)
        weight = tf.Variable(tf.truncated_normal([self.rnn_size, self.class_num], stddev=0.1))
        bais = tf.Variable(tf.constant(0.1, shape=[self.class_num, ]))

        self.prediction = tf.matmul(last, weight) + bais
        self.predict_label = tf.argmax(self.prediction, 1)
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge_all()

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss=self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

    def _extract_axis_1(self, data, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        batch_range = tf.range(self.batch_size)
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)

        return res

    def train(self, sess, batch, keep_prob):
        feed_dict = {
            self.input_data: batch.inputs,
            self.labels: batch.labels,
            self.input_length: batch.input_lengths,
            self.batch_size: len(batch.inputs),
            self.keep_prob: keep_prob
        }
        loss, accuracy, summary_op, _ = sess.run(
            [self.loss, self.accuracy, self.summary_op, self.optimizer],
            feed_dict=feed_dict)

        return loss, accuracy, summary_op

    def train_test(self, sess, batch, keep_prob):
        feed_dict = {
            self.input_data: batch.inputs,
            self.labels: batch.labels,
            self.input_length: batch.input_lengths,
            self.batch_size: len(batch.inputs),
            self.keep_prob: keep_prob
        }
        loss, accuracy, summary_op = sess.run([self.loss, self.accuracy, self.summary_op], feed_dict=feed_dict)
        return loss, accuracy, summary_op

    def predict(self, sess, batch):
        feed_dict = {
            self.input_data: batch.inputs,
            self.input_length: batch.input_lengths,
            self.batch_size: len(batch.inputs),
            self.keep_prob: 1
        }
        pre, pre_label = sess.run([self.prediction, self.predict_label], feed_dict=feed_dict)
        return pre, pre_label
