# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : bi_lstm_train.py
# Time    : 2018/12/19 0019 下午 1:55
"""

import tensorflow as tf
import numpy as np


pb_file_path = './text_rnn_output/1-model.pb'


'单一预测'

def pb_predict(pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            input_x = sess.graph.get_tensor_by_name("input_x:0")
            out_label = sess.graph.get_tensor_by_name("predictions/dimension:0")
            print(out_label)
            out_label = sess.run(out_label,  feed_dict={input_x: np.random.randint(80, size=(80,))[:, np.newaxis].T})
            print(out_label)



#查看节点名
def pb_node_name(pb_file_path):
    with tf.Session() as sess:
        with open(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)
pb_node_name(pb_file_path)
#pb_predict(pb_file_path)