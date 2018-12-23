# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : cnn_model.py
# Time    : 2018/12/19 0019 下午 8:16
"""

import tensorflow as tf

class TextCNN():
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate,  decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size, is_training,is_trace_train=True, class_weight=0):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.class_weight = class_weight
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        if class_weight == 0:
            self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # y [None,num_classes]
        else:
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")


        # 'function'
        # self.logits = self.compute()
        # self.loss = self.loss()
        # self.train = self.train()
        # self.predictions = self.pred()
        # self.accuracy = self.accury()
        self.logits = self.compute()
        self.loss = self.loss()
        self.predictions = self.pred()
        self.accuracy = self.accury()
        if is_trace_train:
            self.train, self.train_summary_op = self.trace_process()
        else:
            self.train = self.train()

    def compute(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        #embeding layer
        with tf.name_scope('embeding'):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size],初始化（-1,1）
            self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
            # [None,sentence_length,embed_size]
            self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)
            # [None,sentence_length,embed_size,1)
        # 2.=====>loop each filter size.
        # for each filter,
        #   do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        # if mulitple_layer_cnn: # this may take 50G memory.
        # else: # this take small memory, less than 2G memory.
        # 2.=====>loop each filter size. for each filter,
        # do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        #h_drop = self.cnn_single_layer()
        h_drop = self.cnn_mutil_layer()
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            self.w = tf.get_variable("w", shape=[self.num_filters_total, self.num_classes],
                                     initializer=self.initializer)  # [embed_size,label_size]
            self.b = tf.get_variable("b", shape=[self.num_classes])  # [label_size]
            print(self.w)
            logits = tf.matmul(h_drop, self.w) + self.b
            print('logits:',logits)
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #print(self.input_y, self.logits)   #self.input_y  为一维向量，每个值都是类别0-num_class

            if self.class_weight == 0:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            else:
                # losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,logits=self.logits)
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

    def cnn_mutil_layer(self):
        pooled_outputs = []
        print("sentence_embeddings_expanded:", self.sentence_embeddings_expanded)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('cnn_multiple_layers' + "-convolution-pooling-%s" % filter_size):
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, self.embed_size, 1],
                                    padding="SAME", name="conv1")
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training, scope='cnn1')
                print(i, "conv1:", conv)
                b1 = tf.get_variable("b1-%s" % filter_size, [self.num_filters])
                print(b1)
                h = tf.nn.relu(tf.nn.bias_add(conv, b1), "relu1")
                print(h)
                # shape:[batch_size,sequence_length,1,num_filters]

                # 2) CNN->BN->relu
                h = tf.reshape(h, [-1, self.sequence_length, self.num_filters, 1])
                # shape:[batch_size,sequence_length,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1,
                                                                       self.num_filters], initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, self.embed_size, 1], padding="SAME", name="conv2")
                # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2), "relu2")
                # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                pooling = tf.nn.max_pool(h, ksize=[1, self.sequence_length, 1, 1], strides=[1, 1, 1, 1],
                                         padding='VALID', name="pool")

                # 3. Max-pooling
                # pooling_max = tf.squeeze(pool)
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling)  # h:[batch_size,sequence_length,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=3)
        # [batch_size,num_filters*len(self.filter_sizes)]
        print("h.concat:", h)
        h_flat = tf.reshape(h, [-1, self.num_filters_total])  # pool_flat
        print("h.flat:", h_flat)
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_flat, keep_prob=self.dropout_keep_prob)
            # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h_drop
    def cnn_single_layer(self):
        pooled_outputs = []
        print('sentence_embeddings_expanded:', self.sentence_embeddings_expanded)
        for i, filter_size in enumerate(self.filter_sizes):
            print(filter_size)
            with tf.variable_scope('CNN_layers' + "-conv-pooling-%s" % filter_size):
                # 1) CONV-BN-RELU layer1
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                # filter：CNN中的卷积核，它要求是一个Tensor，[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="CONV1")
                # conv2d shape:[batch_size,sequence_length - filter_size + 1,1,num_filters] (?, 75, 1, 16)
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training, scope='BN1')
                print(i, "bn1:", conv)
                b1 = tf.get_variable("b1-%s" % filter_size, [self.num_filters])  # bias
                h = tf.nn.relu(tf.nn.bias_add(conv, b1), "RELU1")
                # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                print(i, "relu:", h)
                # 3) Max-pooling  # tf.squeeze 维度为1的去除掉
                # pooled_max:[batch_size,sequence_length,num_filters]
                # pooling_max = tf.squeeze(
                #     tf.nn.max_pool(h, ksize=[1, self.sequence_length, 1, 1], strides=[1, 1, 1, 1],
                #                    padding='VALID', name="POOL"))
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="POOL")
                print(self.sequence_length - filter_size + 1)
                print(i, "pooling:", pooled)  # shape=(?, 1, 1, 16)
                pooled_outputs.append(pooled)
        # concat
        print('pooled_outputs', pooled_outputs)
        h_concat = tf.concat(pooled_outputs, axis=3)  # [batch_size,1,sequence_length,num_filters]
        print("h.concat:", h_concat)
        h_flat = tf.reshape(h_concat, [-1, self.num_filters_total])  # pool_flat
        print("h.flat:", h_flat)
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        print("h.drop:", h_drop)
        return h_drop

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
