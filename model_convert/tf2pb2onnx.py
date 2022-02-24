# -- coding: utf-8 --
# @Time : 2022/2/14
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import tensorflow as tf

# 通过输出的结点名，将与这个结点有关的图及权重保存下来
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                              output_node_names=[input_name.split(':')[0]])
# 写入序列化的 PB 文件，获取带权重的图
with tf.gfile.FastGFile('model.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())

# # 查看input output name
# https://netron.app/

# # 利用tf2onnx将pb转为onnx
# python -m tf2onnx.convert --graphdef model.pb --output model.onnx --inputs 'Placeholder:0' --outputs 'select:0'
