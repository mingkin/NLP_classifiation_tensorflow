# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : bi_lstm_train.py
# Time    : 2018/12/18 0018 下午 5:57
"""

import tensorflow as tf
import numpy as np
import os, sys
import datetime
from utils import utils
from bi_lstm_model import TextRNN
from sklearn.metrics import *
from tensorflow.python.framework import graph_util
from utils.log_util import log

FLAGS=tf.app.flags.FLAGS
'hyparam'
tf.app.flags.DEFINE_integer("num_classes", 2, "number of class")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.app.flags.DEFINE_integer("decay_steps", 10, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_integer("sequence_length", 80, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 2, "epochs")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch_size")
tf.app.flags.DEFINE_integer("class_weight", 10, "class_weight is 0 no weights,other weights.")
tf.app.flags.DEFINE_boolean("is_trace_train", False, "true:trace,false:no trace")
'path'
tf.flags.DEFINE_string("w2v", "./word_vector/word2vec.words", "word2vector_path.")
tf.flags.DEFINE_string("path", "./data/train.txt,./data/test.txt,./data/test.txt", "Data source for  data.")
tf.app.flags.DEFINE_string("ckpt_dir", "text_rnn_output/", "checkpoint location for the model")

log.info('py_dir_name: %s'%sys.argv[0])

def preprocess():
    # Data Preparation
    # ==================================================
    # Load data
    path_data = [i for i in FLAGS.path.split(',')]
    print("Loading data...")
    if FLAGS.class_weight == 0:
        x_train, y_train = utils.read_file_seg(path_data[0])
    else:
        x_train, y_train = utils.read_file_seg_sparse(path_data[0])
    # Build vocabulary
    max_document_length = max([len(x) for x in x_train])
    print(max_document_length)
    print('Loading a Word2vec model...')
    word_2vec = utils.load_word2vec(FLAGS.w2v)  # 加载词向量
    maxlen = FLAGS.sequence_length
    index_dict, word_vectors, x = utils.create_dictionaries(maxlen, word_2vec, x_train)
    print('Embedding weights...')
    vocab_size = FLAGS.embed_size
    word_size, embedding_weights = utils.get_data(index_dict, word_vectors, vocab_dim=vocab_size)
    # test set
    print('Test set ....')
    if FLAGS.class_weight == 0:
        x_test, ytest = utils.read_file_seg(path_data[1])
    else:
        x_test, ytest = utils.read_file_seg_sparse(path_data[1])
    index_dict1, word_vectors1, x_test = utils.create_dictionaries(maxlen, word_2vec, x_test)
    y_test = np.array(ytest)
    log.info("Vocabulary Size: {:d}".format(word_size))
    print('train_x_y_shape', x.shape, y_train.shape)
    print('test_x_y_shape', x_test.shape, y_test.shape)
    print("Vocabulary Size: {:d}".format(word_size))
    return x, y_train, x_test, y_test, embedding_weights


def train(x_train, y_train, embedding_weights, x_dev, y_dev):
    # Training
    # ==================================================
    # Create session
    session_conf = tf.ConfigProto()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        textRNN = TextRNN(num_classes=FLAGS.num_classes, learning_rate=FLAGS.learning_rate,
                          decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
                          sequence_length=FLAGS.sequence_length, vocab_size=embedding_weights.shape[0],
                          embed_size=FLAGS.embed_size, is_trace_train=FLAGS.is_trace_train, class_weight=FLAGS.class_weight)

        #  is or not keep trace
        saver = tf.train.Saver(max_to_keep=FLAGS.num_epochs)
        if FLAGS.is_trace_train:
            if not os.path.exists(os.path.join(FLAGS.ckpt_dir, "tf_log")):
                os.makedirs(os.path.join(FLAGS.ckpt_dir, "tf_log"))
            train_summary_dir = os.path.join(FLAGS.ckpt_dir, "tf_log")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        checkpoint_dir = os.path.abspath(os.path.join(FLAGS.ckpt_dir, "checkpoints"))
        if os.path.exists(checkpoint_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.ckpt_dir, "checkpoint")))
        #Initialize
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        #embeding layer
        assign_pretrained_word_embedding(sess, embedding_weights, textRNN)
        # 3.feed data & training & Generate batches
        batches = utils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        time_str = datetime.datetime.now().isoformat()
        num_batches_per_epoch = int((len(x_train) - 1) / FLAGS.batch_size) + 1
        loss, acc, curr_pre, curr_recall, counter = 0.0, 0.0, 0.0, 0.0, 0
        num = int(x_train.shape[0])
        print('num_batches_per_epoch:', num_batches_per_epoch)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch, y_batch = np.array([list(i) for i in x_batch]), np.array([list(i) for i in y_batch])
            if counter == 0:
                print("trainX:", y_batch.shape)

            if FLAGS.is_trace_train:
                curr_loss, curr_acc, pred, _, summaries, step = sess.run(
                    [textRNN.loss, textRNN.accuracy, textRNN.predictions, textRNN.train, textRNN.train_summary_op,
                     textRNN.global_step],
                    feed_dict={textRNN.input_x: x_batch, textRNN.input_y: y_batch, textRNN.dropout_keep_prob: 0.1})
                train_summary_writer.add_summary(summaries, step)
                print('global_step:', step)
            else:
                curr_loss, curr_acc, pred, _, step = sess.run(
                    [textRNN.loss, textRNN.accuracy, textRNN.predictions, textRNN.train, textRNN.global_step],
                    feed_dict={textRNN.input_x: x_batch, textRNN.input_y: y_batch, textRNN.dropout_keep_prob: 0.1})
                print('global_step:', step)

            if FLAGS.class_weight == 0:
                y_true = [int(i) for i in y_batch]
            else:
                y_true = np.argmax(np.array([list(i) for i in y_batch]), axis=1)
            pre = precision_score(y_true, pred)
            recall = recall_score(y_true, pred)
            loss, counter, acc, pre, recall = loss + curr_loss, counter + 1, acc + curr_acc, \
                                              pre + curr_pre, recall + curr_recall
            if counter % 10 == 0:
                print(time_str, " \t-\tBatch_size %d/%d \t-\tTrain Loss:%.3f \t-\tTrain Accuracy:%.3f"
                                "\t-\tTrain Precision:%.3f \t-\tTrain Recall:%.3f" % (counter, num,
                        loss / float(counter), acc / float(counter), pre / float(counter), recall / float(counter)))
            if counter % num_batches_per_epoch == 0:
                # 4.在测试集上做测试，并报告测试准确率 Test
                evaluate(sess, textRNN, x_dev, y_dev)
                # 5.save model to checkpoint
                epoch = int(counter / num_batches_per_epoch)
                log.info('Epoch: {:d}'.format(epoch))
                print('save model to checkpoint')
                save_path = FLAGS.ckpt_dir+'checkpoint/' + str(epoch) + "-model.ckpt"
                saver.save(sess, save_path, global_step=counter)

                # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['input_x', 'predictions'])
                # 此处务必和前面的输入输出对应上，其他的不用管
                # 写入序列化的 PB 文件
                # 模型的名字是model.pb
                with tf.gfile.FastGFile(FLAGS.ckpt_dir +'checkpoint/' + str(epoch) + '-model.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

def assign_pretrained_word_embedding(sess,embedding_weights,textCNN):
    word_embedding = tf.constant(embedding_weights, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)
    # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("using pre-trained word emebedding.ended...")

def evaluate(sess,textRNN,x,y):
    batches = utils.batch_iter(list(zip(x, y)), FLAGS.batch_size, 1)
    num_batches_per_epoch = int((len(x) - 1) / FLAGS.batch_size) + 1
    loss, acc, curr_pre, curr_recall, counter = 0.0, 0.0, 0.0, 0.0, 0
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        curr_loss, curr_acc, pred, _ = sess.run([textRNN.loss, textRNN.accuracy, textRNN.predictions, textRNN.train],
                                                feed_dict={textRNN.input_x: x_batch, textRNN.input_y: y_batch,
                                                           textRNN.dropout_keep_prob: 0.1})
        if FLAGS.class_weight == 0:
            y_true = [int(i) for i in y_batch]
        else:
            y_true = np.argmax(np.array([list(i) for i in y_batch]), axis=1)
        pre = precision_score(y_true, pred)
        recall = recall_score(y_true, pred)
        loss, counter, acc, pre, recall = loss + curr_loss, counter + 1, acc + curr_acc, pre + curr_pre, recall + curr_recall
        if counter % num_batches_per_epoch == 0:
            log.info("\t-\tBatch_size %d\t-\tTest Loss:%.3f\t-\tTest Accuracy:%.3f\t-\t"
                     "Test Precision:%.3f\t-\tTest Recall:%.3f " % (counter,loss / float(counter), acc / float(counter),
                                                                pre / float(counter), recall / float(counter)))


def main(_):
    x_train, y_train, x_dev, y_dev, embedding_weights, = preprocess()
    train(x_train, y_train, embedding_weights, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()