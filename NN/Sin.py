import tensorflow as tf
import numpy as np


class SinNN:
    def __init__(self):
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.network, self.loss = self.build()

    def build(self):
        self.w0 = tf.Variable(tf.random.normal(shape=[1, 256], stddev=0.05, dtype=tf.float32))
        self.b0 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))
        self.w1 = tf.Variable(tf.random.normal(shape=[256, 256], stddev=0.05, dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))
        self.w2 = tf.Variable(tf.random.normal(shape=[256, 1], stddev=0.01, dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32))

        h0 = tf.matmul(self.features, self.w0) + self.b0
        h0 = tf.nn.relu(h0)
        h1 = tf.matmul(h0, self.w1) + self.b1
        h1 = tf.nn.relu(h1)
        h2 = tf.matmul(h1, self.w2) + self.b2

        cost = tf.abs(h2 - self.labels)
        loss = tf.reduce_mean(cost)
        return h2, loss

    def save(self, sess, result_folder):
        w0 = sess.run(self.w0)
        w0.tofile(result_folder + 'w0.bin')
        del w0
        w1 = sess.run(self.w1)
        w1.tofile(result_folder + 'w1.bin')
        del w1
        w2 = sess.run(self.w2)
        w2.tofile(result_folder + 'w2.bin')
        del w2
        b0 = sess.run(self.b0)
        b0.tofile(result_folder + 'b0.bin')
        del b0
        b1 = sess.run(self.b1)
        b1.tofile(result_folder + 'b1.bin')
        del b1
        b2 = sess.run(self.b2)
        b2.tofile(result_folder + 'b2.bin')
        del b2
        pass
