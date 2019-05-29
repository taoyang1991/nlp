# encoding=utf-8
import datetime

import os

from data_utils import *
from model import RNNModel
import tensorflow as tf
from utils import get_logger, create_path, export_model
import sys
import math
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")


class ModelUsage():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.root_path = os.path.join(self.FLAGS.root_path)
        self.is_train = self.FLAGS.train
        self.embedding_size = self.FLAGS.embedding_size
        self.rnn_size = self.FLAGS.rnn_size
        self.class_num = self.FLAGS.class_num
        self.learning_rate = self.FLAGS.learning_rate
        self.dropout = self.FLAGS.dropout
        self.batch_size = self.FLAGS.batch_size
        self.epoch_num = self.FLAGS.epoch_num
        self.epoch_step = self.FLAGS.epoch_step
        self.model_dir = os.path.join(self.root_path, self.FLAGS.model_dir)
        self.model_path = os.path.join(self.root_path, self.FLAGS.model_path)
        self.summary_path = os.path.join(self.root_path, self.FLAGS.summary_path)
        self.vocab_file = os.path.join(self.root_path, self.FLAGS.vocab_file)
        self.log_path = os.path.join(self.root_path, self.FLAGS.log_path)
        self.logfile_path = os.path.join(self.log_path, "train.log")
        self.train_data_path = os.path.join(self.root_path, self.FLAGS.train_data_path)
        self.test_data_path = os.path.join(self.root_path, self.FLAGS.test_data_path)
        self.is_export = self.FLAGS.export

    def training(self):
        vocab_to_id = get_vocab_to_id(self.train_data_path, self.vocab_file, False)
        logdir = os.path.join(self.summary_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/")
        self.vocab_size = len(vocab_to_id)

        create_path(self.log_path)
        logger = get_logger(self.logfile_path)

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
            summary_writer.flush()
            rnn_model = RNNModel(self.rnn_size, self.embedding_size, self.class_num, self.vocab_size,
                                 self.learning_rate, self.model_path)

            test_data_generator = SentenceGenerator(self.test_data_path)
            testBatchManage = BatchManager(test_data_generator, 0, vocab_to_id)
            test_data = testBatchManage.get_all_data_to_batch()

            sess.run(tf.global_variables_initializer())
            current_step = 0
            for e in range(self.epoch_num):
                logger.info("Epoch num: " + str(e + 1) + "\n")
                print("Epoch num: " + str(e + 1) + "\n")
                train_data_generator = SentenceGenerator(self.train_data_path)
                trainBatchManage = BatchManager(train_data_generator, self.batch_size, vocab_to_id)
                for batchs in trainBatchManage.getBatches():
                    current_step += 1

                    loss, accuracy, summary_op = rnn_model.train(sess, batchs, self.dropout)
                    if current_step % self.epoch_step == 0:
                        loss_test, accuracy_test, _ = rnn_model.train_test(sess, test_data, 1.0)
                        logger.info("loss:" + str(loss_test) + " accuracy:" + str(accuracy_test) + "\n")
                        print("loss:" + str(loss_test) + " accuracy:" + str(accuracy_test) + "\n")
                        summary_writer.add_summary(summary_op, current_step)
                        rnn_model.saver.save(sess, self.model_path, global_step=current_step)

    def predicting(self):
        vocab_to_id = get_vocab_to_id(self.train_data_path, self.vocab_file, False)
        data_helper = DataHelper(vocab_to_id)

        reader = open("data/predict_data", 'r')
        writer = open("data/res", 'w')
        dishs = []
        file_data = []
        for line in reader.readlines():
            line = line.strip().decode("utf-8")
            line_split = line.split("\t")
            if len(line_split) != 2:
                continue
            type,dish_name = line_split
            dishs.append(dish_name)
            file_data.append(line_split)

        batch = data_helper.create_prediction_batch(dishs)
        with tf.Session() as sess:
            cnn_model = RNNModel(self.rnn_size, self.embedding_size, self.class_num, len(vocab_to_id),
                                 self.learning_rate, self.model_path)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            cnn_model.saver.restore(sess, ckpt.model_checkpoint_path)
            prediction, pre_label = cnn_model.predict(sess, batch)
            pre_pre = sess.run(tf.nn.softmax(prediction))
            print pre_label, pre_pre, prediction
            for idx, sub_review_lable in enumerate(pre_label):
                writer.write("{}\t{}\t{}\n".format(file_data[idx][0], file_data[idx][1], sub_review_lable))

    def predicting_1(self):
        vocab_to_id = get_vocab_to_id(self.train_data_path, self.vocab_file, False)
        data_helper = DataHelper(vocab_to_id)

        dishs = [
           u"网上的口碑什么的蛮好的一家店 专门打了个电话让这边的师傅上门帮我们家的小宝宝理了一个头发"
        ]
        batch = data_helper.create_prediction_batch(dishs)
        with tf.Session() as sess:
            cnn_model = RNNModel(self.rnn_size, self.embedding_size, self.class_num, len(vocab_to_id),
                                 self.learning_rate, self.model_path)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            cnn_model.saver.restore(sess, ckpt.model_checkpoint_path)
            prediction, pre_label = cnn_model.predict(sess, batch)
            pre_pre = sess.run(tf.nn.softmax(prediction))
            print pre_label, pre_pre, prediction
            for idx, sub_review_lable in enumerate(pre_label):
                print "{}\t{}".format(dishs[idx], data_helper.get_cats_name(sub_review_lable))

    def predicting_2(self):
        vocab_to_id = get_vocab_to_id(self.train_data_path, self.vocab_file, False)
        data_helper = DataHelper(vocab_to_id)
        data_generator = SentenceGenerator("data/other_data")
        batchManage = BatchManager(data_generator, self.batch_size, vocab_to_id)
        writer = open("data/res_other", "w")
        with tf.Session() as sess:
            models = RNNModel(self.rnn_size, self.embedding_size, self.class_num, len(vocab_to_id),
                              self.learning_rate, self.model_path)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            models.saver.restore(sess, ckpt.model_checkpoint_path)

            for batchs in batchManage.getBatches():
                prediction,pre_label= models.predict(sess, batchs)
                for sub_review_lable in pre_label:
                    writer.write(str(sub_review_lable)+"\n")


    def exporting_model(self):
        vocab_to_id = get_vocab_to_id(self.train_data_path, self.vocab_file, False)
        with tf.Session() as sess:
            rnn_model = RNNModel(self.rnn_size, self.embedding_size, self.class_num, len(vocab_to_id),
                                 self.learning_rate, self.model_path)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            rnn_model.saver.restore(sess, ckpt.model_checkpoint_path)
            export_model(sess, rnn_model, "export/", "1", vocab_to_id)

    def run(self):
        if self.is_export:
            self.exporting_model()
        elif self.is_train:
            self.training()
        else:
            self.predicting_1()
