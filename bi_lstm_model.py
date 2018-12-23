# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : bi_lstm_model.py
# Time    : 2018/12/18 0018 上午 9:09
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class TextRNN():
    def __init__(self, num_classes, learning_rate,  decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, is_trace_train=True, class_weight=0, hidden_size=200):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        if class_weight == 0:
            self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # y [None,num_classes]
        else:
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.class_weight = class_weight

        'function'
        self.logits = self.compute()
        self.loss = self.loss()
        self.predictions = self.pred()
        self.accuracy = self.accury()
        if is_trace_train:
            self.train, self.train_summary_op = self.trace_process()
        else:
            self.train = self.train()


    def compute(self):
        # Embeding layer
        with tf.name_scope('Embeding'):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                                                      initializer=self.initializer)
            #[vocab_size,embed_size]
            self.W = tf.get_variable("W", shape=[self.hidden_size * 2, self.num_classes],
                        initializer=self.initializer)                #[embed_size*2,label_size]
            self.b = tf.get_variable("b", shape=[self.num_classes])  # [num_classes,]
        # 1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        #print('embedded_words:', self.embedded_words, self.W, self.b)
        # shape:[None,sentence_length,embed_size]
        # 2. Bi-lstm layer
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
            # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
            # output: A tuple (outputs, output_states)
            # where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        # [batch_size,sequence_length,embed_size] #creates a dynamic bidirectional recurrent neural network
        #print("outputs:===>", outputs)
        #3.concat output 将两个cell的outputs进行拼接
        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        #print(output_rnn)
        # self.output_rnn_last=tf.reduce_mean(output_rnn,axis=1) #[batch_size,hidden_size*2] 归一化
        # 3. Second LSTM layer
        rnn_cell = rnn.BasicLSTMCell(self.hidden_size * 2)
        if self.dropout_keep_prob is not None:
            rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
        _, final_state_c_h = tf.nn.dynamic_rnn(rnn_cell, output_rnn, dtype=tf.float32)
        final_state = final_state_c_h[1]     #h 为输出  output对应output门
        #print(final_state_c_h)
        # 4 .FC layer
        output = tf.layers.dense(final_state, self.hidden_size * 2, activation=tf.nn.tanh)
        # 5.logits(use linear layer)
        with tf.name_scope("output"):
            logits = tf.matmul(output, self.W) + self.b  # [batch_size,num_classes]
        #print(logits)
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #print(self.input_y, self.logits)   #self.input_y  为一维向量，每个值都是类别0-num_class
            #losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.logits)
            if self.class_weight == 0:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            else:
                losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y,
                                                        logits=self.logits, pos_weight=self.class_weight)
            #print("1.losses:", losses) # pos_weight is 1 weights
            loss = tf.reduce_mean(losses)
            #print("2.loss.loss:", loss)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate  # 查看衰减学习率
        train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                learning_rate=learning_rate, optimizer="Adam")
        return train_op

    def accury(self):
        if self.class_weight == 0:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        else:
            correct_prediction = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        return accuracy
    def pred(self):
        predictions = tf.argmax(self.logits, axis=1, name="predictions")
        return predictions

    def trace_process(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate  # 查看衰减学习率
        # Define Training procedure
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        return train_op, train_summary_op



