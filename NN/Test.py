import tensorflow as tf
import numpy as np

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# i = tf.constant(0)
# a = tf.constant([[[0., 1, 2]], [[0.5, 1.5, 2.5]]])
# b = tf.constant([1, 0.7])
# p = tf.constant([[[0.4], [0.4], [0.4]], [[0.6], [0.6], [0.6]]])
# print(a.shape)

w = tf.placeholder(dtype=tf.float32, shape=[2, 3, 4])
v = tf.placeholder(dtype=tf.float32, shape=[None, 2, 1, 1])


with tf.Session(config=tf_config) as sess:
    feed_dict = {
        w: [[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8]]],
        v: [[[[1]], [[1]]], [[[0.5]], [[0.6]]]]
    }
    c = sess.run(tf.reduce_sum(tf.multiply(w, v),  axis=1), feed_dict=feed_dict)

    print(c)
