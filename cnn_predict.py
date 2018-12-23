# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : cnn_predict.py
# Time    : 2018/12/19 0019 下午 8:17
"""



import tensorflow as tf
import numpy as np


pb_file_path = './text_cnn_output/2-model.pb'


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
            output_x = sess.graph.get_tensor_by_name("predictions/dimension:0")
            print(output_x)
            output_x = sess.run(output_x, feed_dict={input_x: np.random.randint(80, size=(80,))[:, np.newaxis].T})
            print(output_x)



#查看节点名
def pb_node_name(pb_file_path):
    with tf.Session() as sess:
        with open(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)
pb_node_name(pb_file_path)
pb_predict(pb_file_path)





