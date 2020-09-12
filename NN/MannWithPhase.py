import tensorflow as tf
import numpy as np

class MannWithPhase:
    def __init__(self, xdim, ydim, nslice, hdim, gate_hdim):
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, xdim], name='Features')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ydim], name='Labels')
        self.phase = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Phase')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='KeepBlob')
        self.hdim = hdim
        self.nslice = nslice
        self.xdim = xdim
        self.ydim = ydim
        self.gate_hdim = gate_hdim
        self.gate_xdim = 2

        self.gate_network = self.build_gate_network()
        self.network, self.loss = self.build()

    def build(self):
        self.w0 = tf.Variable(tf.random_normal(shape=[self.nslice, self.xdim, self.hdim], stddev=0.5, dtype=tf.float32), name='W0')
        self.b0 = tf.Variable(tf.zeros(shape=[self.nslice, self.hdim], dtype=tf.float32), name='B0')
        self.w1 = tf.Variable(tf.random_normal(shape=[self.nslice, self.hdim, self.hdim], stddev=0.6, dtype=tf.float32), name='W1')
        self.b1 = tf.Variable(tf.zeros(shape=[self.nslice, self.hdim], dtype=tf.float32), name='B1')
        self.w2 = tf.Variable(tf.random_normal(shape=[self.nslice, self.hdim, self.ydim], stddev=0.1, dtype=tf.float32), name='W2')
        self.b2 = tf.Variable(tf.zeros(shape=[self.nslice, self.ydim], dtype=tf.float32), name='B2')

        blend0 = tf.expand_dims(self.gate_network, -1)
        blend1 = tf.expand_dims(blend0, -1)

        w0 = tf.reduce_sum(tf.multiply(self.w0, blend1), axis=1)
        w1 = tf.reduce_sum(tf.multiply(self.w1, blend1), axis=1)
        w2 = tf.reduce_sum(tf.multiply(self.w2, blend1), axis=1)
        b0 = tf.reduce_sum(tf.multiply(self.b0, blend0), axis=1)
        b1 = tf.reduce_sum(tf.multiply(self.b1, blend0), axis=1)
        b2 = tf.reduce_sum(tf.multiply(self.b2, blend0), axis=1)

        hi = self.features
        hi = tf.expand_dims(hi, 1)
        hi = tf.nn.dropout(hi, keep_prob=self.keep_prob)

        b0 = tf.expand_dims(b0, 1)
        h0 = tf.matmul(hi, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob=self.keep_prob)

        b1 = tf.expand_dims(b1, 1)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

        b2 = tf.expand_dims(b2, 1)
        h2 = tf.matmul(h1, w2) + b2
        h2 = tf.squeeze(h2)

        cost = tf.square(h2 - self.labels)
        loss = tf.reduce_mean(cost)
        return h2, loss

    def save(self, sess, result_folder):
        # Export gate network weight and bias
        gw0 = sess.run(self.gw0)
        gw0.tofile(result_folder + 'gw0.bin')
        del gw0
        gw1 = sess.run(self.gw1)
        gw1.tofile(result_folder + 'gw1.bin')
        del gw1
        gw2 = sess.run(self.gw2)
        gw2.tofile(result_folder + 'gw2.bin')
        del gw2
        gb0 = sess.run(self.gb0)
        gb0.tofile(result_folder + 'gb0.bin')
        del gb0
        gb1 = sess.run(self.gb1)
        gb1.tofile(result_folder + 'gb1.bin')
        del gb1
        gb2 = sess.run(self.gb2)
        gb2.tofile(result_folder + 'gb2.bin')
        del gb2

        for i in range(self.nslice):
            w0 = sess.run(tf.nn.embedding_lookup(self.w0, i))
            w0.tofile(result_folder + 'w0_%i.bin' % i)
            del w0
            w1 = sess.run(tf.nn.embedding_lookup(self.w1, i))
            w1.tofile(result_folder + 'w1_%i.bin' % i)
            del w1
            w2 = sess.run(tf.nn.embedding_lookup(self.w2, i))
            w2.tofile(result_folder + 'w2_%i.bin' % i)
            del w2

            b0 = sess.run(tf.nn.embedding_lookup(self.b0, i))
            b0.tofile(result_folder + 'b0_%i.bin' % i)
            del b0
            b1 = sess.run(tf.nn.embedding_lookup(self.b1, i))
            b1.tofile(result_folder + 'b1_%i.bin' % i)
            del b1
            b2 = sess.run(tf.nn.embedding_lookup(self.b2, i))
            b2.tofile(result_folder + 'b2_%i.bin' % i)
            del b2

    def build_gate_network(self):
        self.gw0 = tf.Variable(tf.random_normal(shape=[self.gate_xdim, self.gate_hdim], stddev=0.1, dtype=tf.float32), name='GW0')
        self.gb0 = tf.Variable(tf.zeros(shape=[self.gate_hdim], dtype=tf.float32), name='GB0')
        self.gw1 = tf.Variable(tf.random_normal(shape=[self.gate_hdim, self.gate_hdim], stddev=0.1, dtype=tf.float32), name='GW1')
        self.gb1 = tf.Variable(tf.zeros(shape=[self.gate_hdim], dtype=tf.float32), name='GB1')
        self.gw2 = tf.Variable(tf.random_normal(shape=[self.gate_hdim, self.nslice], stddev=0.1, dtype=tf.float32), name='GW2')
        self.gb2 = tf.Variable(tf.zeros(shape=[self.nslice], dtype=tf.float32), name='GB2')

        hi = tf.concat([tf.sin(self.phase * 2 * 3.1415926), tf.cos(self.phase * 2 * 3.1415926)], axis=-1)
        # hi = self.phase

        h0 = tf.matmul(hi, self.gw0) + self.gb0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob=self.keep_prob)

        h1 = tf.matmul(h0, self.gw1) + self.gb1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

        h2 = tf.matmul(h1, self.gw2) + self.gb2
        h2 = tf.nn.softmax(h2)

        return h2
