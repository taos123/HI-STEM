# -*- coding: UTF-8 -*-

# Author : sun tao
# Time   : 2019/4/12
# E-mail : suntao@ict.ac.cn

import tensorflow as tf
import numpy as np

import datetime


from tul_attention.data_read import DataReader
from tul_attention.MacroF1 import MacroF1
from tul_attention.attention import attention

from tul_attention.config import Config


def RNN(x, weights, biases, keep_prob, n_hidden, seq_length):

    # print("原始x维度")
    # print(x.shape)
    x = tf.transpose(x, [1, 0, 2])  # transform position for different dimension

    # print("time major 后x的维度")
    # print(x.shape)
    # forward, state_is_tuple=True, return c_state and m_state
    fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
    # use dropout
    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # use dropout
    bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.5, state_is_tuple=True)
    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)

    # bidirectional LSTM
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
    # dynamic rnn
    (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32, time_major=True, sequence_length=seq_length)  # ,dtype=tf.float32,time_major=True  ,initial_state_fw=istate_fw,initial_state_bw=istate_bw

    # print("双向lstm outputs的维度")
    # # print(size(outputs))
    # ???
    # new_outputs = tf.concat(outputs, 2)

    new_outputs = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

    # new_outputs = tf.nn.dropout(new_outputs, keep_prob=0.5)

    # print("拼接output后的维度")
    # print(new_outputs.shape)
    # # matrix multiplication
    #
    # print('三个参数维度分别为')
    # print(new_outputs[-1].shape)
    # print(weights['out'].shape)
    # print(biases['out'].shape)

    val = tf.matmul(new_outputs, weights['out'])+biases['out']
    #val=tf.add(tf.add(tf.matmul(fw_output[-1],weights['out']),tf.matmul(fb_output[-1],weights['out'])), biases['out'])

    # print("rnn计算值维度")
    # print(val.shape)
    return val


class BiLSTM:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.keep_prob = tf.placeholder(tf.float32)  # rate = 1 - keep_prob

        self.n_input = Config.input_size  # embedding size
        self.n_hidden = Config.hidden_size  # hidden size

        self.n_classes = Config.class_number  # class number

        self.x = tf.placeholder("float", [self.batch_size, None, self.n_input])  #

        self.y_out = tf.placeholder("float", [self.batch_size, self.n_classes])

        self.weights = {
            'out': tf.Variable(tf.random_normal([2 * self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.pred = RNN(self.x, self.weights, self.biases, self.keep_prob, self.n_hidden, self.seq_length)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y_out))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  #
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_out, 1))  # 1
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.sess = tf.Session()

        self.reader = DataReader()

    def save_model(self, model_name):
        saver = tf.train.Saver()
        saver.save(self.sess, model_name)

    def restore(self, model_name):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_name)

    def init_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train_(self, batch_size=1):
        while True:
            x_in, y_in, seq_len = self.reader.read_train_data_mixup(batch_size)
            # x_in, y_in, seq_len = self.reader.read_train_data(batch_size)

            if x_in is None:
                break

            self.sess.run(self.optimizer, feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 0.5, self.seq_length: seq_len})
            # print("complete one sess")
            # a = input("hello suntao")
        print("complete epoch training")

    def test_(self, batch_size=1):

        acc_top1 = 0
        acc_top5 = 0

        macrof1_dict = dict()
        test_count = 0

        while True:

            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            for batch_num in range(0, batch_size):
                for index in range(0, 1):
                    true_uesr = y_in[batch_num].index(1)
                    macrof1_dict[true_uesr] = MacroF1(true_uesr)

        while True:

            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            nowVec = self.sess.run(self.pred, feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 1.0, self.seq_length: seq_len})  #

            predictList = np.argpartition(a=-nowVec, kth=5)
            predictList_top1 = np.argpartition(a=-nowVec, kth=1)

            for batch_num in range(0, batch_size):
                for index in range(0, 5):
                    if predictList[batch_num][index] == y_in[batch_num].index(1):
                        acc_top5 += 1
                        break



            for batch_num in range(0, batch_size):
                for index in range(0, 1):

                    test_count += 1

                    predict_user = predictList_top1[batch_num][index]
                    true_uesr = y_in[batch_num].index(1)

                    if predict_user == true_uesr:
                        macrof1_dict[true_uesr].TP += 1
                    else:
                        macrof1_dict[true_uesr].FN += 1
                        if predict_user in macrof1_dict.keys():
                            macrof1_dict[predict_user].FP += 1


                    if predictList_top1[batch_num][index] == y_in[batch_num].index(1):
                        acc_top1 += 1
                        break

        print(acc_top1 / test_count)
        print(acc_top5 / test_count)

        TP = sum([macrof1_dict[key].TP for key in macrof1_dict.keys()])
        FP = sum([macrof1_dict[key].FP for key in macrof1_dict.keys()])
        FN = sum([macrof1_dict[key].FN for key in macrof1_dict.keys()])

        print(TP, FP, FN)

        microP = TP / (TP + FP)
        microR = TP / (TP + FN)

        microf1 = 2 * (microP * microR) / (microP + microR)

        macrof1 = np.mean([macrof1_dict[key].get_marcof1() for key in macrof1_dict.keys()])

        macroP = np.mean([macrof1_dict[key].get_P() for key in macrof1_dict.keys()])
        macroR = np.mean([macrof1_dict[key].get_R() for key in macrof1_dict.keys()])

        print(microP, microR, microf1, macrof1)

        print(macroP, macroR, 2 * (macroP * macroR) / (macroP + macroR))


if __name__ == "__main__":

    learning_rate = 0.00095
    batch_size = 1
    iterations = 30

    model_name = Config.model_name
    results_file_name = Config.results_file_name
    bi_lstm = BiLSTM(learning_rate, batch_size)

    # bi_lstm.restore(model_name)
    bi_lstm.init_model()

    for i in range(0, iterations):
        print(datetime.datetime.now())
        print("epoch number: ", i)
        bi_lstm.train_(batch_size=batch_size)
        bi_lstm.test_(batch_size=batch_size)
        bi_lstm.save_model(model_name)
        # bi_lstm.save_results(results_file_name)


