import tensorflow as tf
import numpy as np


class ConvolutionalEncoderDecoder:
    def __init__(self, hdim, wdim, ydim):
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, hdim, wdim, 1])   # [N, H, W, C] W is temporal
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ydim, 1, 1])        # [N, H, W, C]

        self.xMean = tf.placeholder(dtype=tf.float32, shape=[hdim, wdim, 1])
        self.yMean = tf.placeholder(dtype=tf.float32, shape=[ydim, 1, 1])
        self.xStd = tf.placeholder(dtype=tf.float32, shape=[hdim, wdim, 1])
        self.yStd = tf.placeholder(dtype=tf.float32, shape=[ydim, 1, 1])
        # self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.hdim = hdim
        self.wdim = wdim
        self.ydim = ydim

        self.network, self.loss = self.build()

    def build(self):
        feature_map = self.build_encoder(self.features)
        h = self.build_decoder(feature_map)
        # eps = tf.random_normal(tf.shape(feature_map))
        # kld = tf.reduce_sum(tf.multiply(feature_map, tf.log(feature_map / eps)))
        # ce = tf.nn.softmax_cross_entropy_with_logits(labels=eps, logits=feature_map)
        # zeros = tf.zeros(tf.shape(feature_map))
        # mse = tf.square(feature_map - zeros)
        # self.mse = tf.reduce_sum(mse)
        cost = tf.square(h - self.labels)
        loss = tf.reduce_mean(cost)  # + tf.reduce_sum(mse)
        return h, loss

    def save(self, sess, result_folder):
        ew0 = sess.run(self.ew0)
        ew0.astype(np.float32).tofile(result_folder + 'ew0.bin')
        del ew0
        ew1 = sess.run(self.ew1)
        ew1.astype(np.float32).tofile(result_folder + 'ew1.bin')
        del ew1
        ew2 = sess.run(self.ew2)
        ew2.astype(np.float32).tofile(result_folder + 'ew2.bin')
        del ew2
        # ew3 = sess.run(self.ew3)
        # ew3.astype(np.float32).tofile(result_folder + 'ew3.bin')
        # del eb0
        eb0 = sess.run(self.eb0)
        eb0.astype(np.float32).tofile(result_folder + 'eb0.bin')
        del eb0
        eb1 = sess.run(self.eb1)
        eb1.astype(np.float32).tofile(result_folder + 'eb1.bin')
        del eb1
        eb2 = sess.run(self.eb2)
        eb2.astype(np.float32).tofile(result_folder + 'eb2.bin')
        del eb2

        dw0 = sess.run(self.dw0)
        dw0.astype(np.float32).tofile(result_folder + 'dw0.bin')
        del dw0
        db0 = sess.run(self.db0)
        db0.astype(np.float32).tofile(result_folder + 'db0.bin')
        del db0
        dw1 = sess.run(self.dw1)
        dw1.astype(np.float32).tofile(result_folder + 'dw1.bin')
        del dw1
        db1 = sess.run(self.db1)
        db1.astype(np.float32).tofile(result_folder + 'db1.bin')
        del db1
        dw2 = sess.run(self.dw2)
        dw2.astype(np.float32).tofile(result_folder + 'dw2.bin')
        del dw2
        db2 = sess.run(self.db2)
        db2.astype(np.float32).tofile(result_folder + 'db2.bin')
        del db2

    def build_encoder(self, input):
        self.ew0 = tf.Variable(tf.random.normal(shape=[self.hdim, 12, 1, 64], dtype=tf.float32))        # [H, W, IC, OC]
        self.eb0 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))
        self.ew1 = tf.Variable(tf.random.normal(shape=[64, 24,  1, 24], dtype=tf.float32))
        self.eb1 = tf.Variable(tf.zeros(shape=[24], dtype=tf.float32))
        self.ew2 = tf.Variable(tf.random.normal(shape=[24, 16, 1, 8], dtype=tf.float32))
        self.eb2 = tf.Variable(tf.zeros(shape=[8], dtype=tf.float32))

        # [N, 93, 120, 1]
        h0 = tf.nn.conv2d(input, self.ew0, [1, 2, 2, 1], 'VALID')
        h0 = h0 + self.eb0
        # [N, 1, 55, 64]
        h0 = tf.transpose(h0, [0, 3, 2, 1])
        # [N, 55, 64, 1] -> [N, 64, 55, 1]
        mean, var = tf.nn.moments(h0, axes=[0])
        h0 = tf.nn.batch_normalization(h0, mean, var, None, None, 1e-8)
        h0 = tf.nn.elu(h0)
        # h0 = tf.squeeze(h0, axis=1)
        # h0 = tf.expand_dims(h0, axis=-1)

        h1 = tf.nn.conv2d(h0, self.ew1, [1, 2, 2, 1], 'VALID')
        h1 = h1 + self.eb1
        # [N, 1, 16, 24]
        h1 = tf.transpose(h1, [0, 3, 2, 1])
        # [N, 16, 24, 1] -> [N, 24, 16, 1]
        mean, var = tf.nn.moments(h1, axes=[0])
        h1 = tf.nn.batch_normalization(h1, mean, var, None, None, 1e-8)
        h1 = tf.nn.elu(h1)
        # h1 = tf.squeeze(h1, axis=1)
        # h1 = tf.expand_dims(h1, axis=-1)

        h2 = tf.nn.conv2d(h1, self.ew2, [1, 1, 1, 1], 'VALID')
        h2 = h2 + self.eb2
        mean, var = tf.nn.moments(h2, axes=[0])
        h2 = tf.nn.batch_normalization(h2, mean, var, None, None, 1e-8)
        # h2 = tf.nn.tanh(h2)
        # [N, 1, 1, 8]
        # h2 = tf.nn.softmax(h2)

        return h2

    def build_decoder(self, feature_map):
        self.dw0 = tf.Variable(tf.random.normal(shape=[128, 1, 1, 8], dtype=tf.float32))        # [H, W, OC, IC]
        self.db0 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))
        self.dw1 = tf.Variable(tf.random.normal(shape=[64, 1, 1, 128], dtype=tf.float32))
        self.db1 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))
        self.dw2 = tf.Variable(tf.random.normal(shape=[93, 1, 1, 64], dtype=tf.float32))
        self.db2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32))

        batch_size = tf.shape(self.features)[0]

        h0 = tf.nn.conv2d_transpose(feature_map, self.dw0, [batch_size, 128, 1, 1], [1, 1, 1, 1], 'VALID')
        # [N, 128, 1, 1]
        h0 = tf.transpose(h0, [0, 2, 3, 1])
        h0 = h0 + self.db0
        h0 = tf.nn.elu(h0)
        # [N, 1, 1, 128]

        h1 = tf.nn.conv2d_transpose(h0, self.dw1, [batch_size, 64, 1, 1], [1, 1, 1, 1], 'VALID')
        # [N, 64, 1, 1]
        h1 = tf.transpose(h1, [0, 2, 3, 1])
        h1 = h1 + self.db1
        h1 = tf.nn.elu(h1)

        # [N, 1, 1, 64]
        h2 = tf.nn.conv2d_transpose(h1, self.dw2, tf.shape(self.labels), [1, 1, 1, 1], 'VALID')
        h2 = h2 + self.db2
        return h2
