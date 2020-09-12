import tensorflow as tf
import numpy as np


class VariationalEncoderDecoder:
    def __init__(self, hdim, wdim, ydim):
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, hdim * wdim])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ydim])
        # self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.hdim = hdim
        self.wdim = wdim
        self.ydim = ydim

        self.network, self.loss = self.build()

    def build(self):
        # feature_map = self.build_encoder(self.features)
        mean, logv = self.build_encoder(self.features)
        feature_map = self.Gaussion(mean, logv)
        h = self.build_decoder(feature_map)
        # h = tf.squeeze(h)
        # cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=h)
        cost = tf.square(h - self.labels)
        kl = 0.5 * tf.reduce_sum(tf.exp(logv) + mean ** 2 - 1.0 - logv, axis=1)
        loss = tf.reduce_mean(tf.reduce_sum(cost, 1) + kl)

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
        ew3 = sess.run(self.ew3)
        ew3.astype(np.float32).tofile(result_folder + 'ew3.bin')
        del ew3
        ew4 = sess.run(self.ew4)
        ew4.astype(np.float32).tofile(result_folder + 'ew4.bin')
        del ew4
        eb0 = sess.run(self.eb0)
        eb0.astype(np.float32).tofile(result_folder + 'eb0.bin')
        del eb0
        eb1 = sess.run(self.eb1)
        eb1.astype(np.float32).tofile(result_folder + 'eb1.bin')
        del eb1
        eb2 = sess.run(self.eb2)
        eb2.astype(np.float32).tofile(result_folder + 'eb2.bin')
        del eb2
        eb3 = sess.run(self.eb3)
        eb3.astype(np.float32).tofile(result_folder + 'eb3.bin')
        del eb3
        eb4 = sess.run(self.eb3)
        eb4.astype(np.float32).tofile(result_folder + 'eb4.bin')
        del eb4

        dw0 = sess.run(self.w0)
        dw0.astype(np.float32).tofile(result_folder + 'dw0.bin')
        del dw0
        db0 = sess.run(self.b0)
        db0.astype(np.float32).tofile(result_folder + 'db0.bin')
        del db0
        dw1 = sess.run(self.w1)
        dw1.astype(np.float32).tofile(result_folder + 'dw1.bin')
        del dw1
        db1 = sess.run(self.b1)
        db1.astype(np.float32).tofile(result_folder + 'db1.bin')
        del db1
        dw2 = sess.run(self.w2)
        dw2.astype(np.float32).tofile(result_folder + 'dw2.bin')
        del dw2
        db2 = sess.run(self.b2)
        db2.astype(np.float32).tofile(result_folder + 'db2.bin')
        del db2

    def build_encoder(self, input):
        '''
        self.kernel0 = tf.Variable(tf.random.normal([10, 3, 15], stddev=0.5, dtype=tf.float32))
        self.bias0 = tf.Variable(tf.zeros([15], dtype=tf.float32))
        self.kernel1 = tf.Variable(tf.random.normal([15, 15, 64], stddev=0.5, dtype=tf.float32))
        self.bias1 = tf.Variable(tf.zeros([64], dtype=tf.float32))
        self.kernel2 = tf.Variable(tf.random.normal([14, 64, 1], stddev=0.5, dtype=tf.float32))
        self.bias2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

        # (w - kernel_size)/stride + 1
        h0 = tf.nn.conv1d(self.features, self.kernel0, 10, 'VALID')
        h0 = h0 + self.bias0
        h0 = tf.nn.leaky_relu(h0)
        h0 = tf.nn.avg_pool(h0, [], [], 'VALID')

        h1 = tf.nn.conv1d(h0, self.kernel1, 1, 'VALID')
        h1 = h1 + self.bias1
        h1 = tf.nn.leaky_relu(h1)

        h2 = tf.nn.conv1d(h1, self.kernel2, 1, 'VALID')
        h2 = h2 + self.bias2
        h2 = tf.nn.softmax(h2)      # Shape = [None, 4]
        '''
        '''
        self.kernel0 = tf.Variable(tf.random.normal([self.hdim, 45, 1, 32], stddev=0.5, dtype=tf.float32))  # [H, W, IC, OC]
        self.bias0 = tf.Variable(tf.zeros([32], dtype=tf.float32))
        self.kernel1 = tf.Variable(tf.random.normal([76, 16, 1, 32], stddev=0.5, dtype=tf.float32))
        self.bias1 = tf.Variable(tf.zeros([32], dtype=tf.float32))
        self.kernel2 = tf.Variable(tf.random.normal([17, 9, 1, 1], stddev=0.5, dtype=tf.float32))
        self.bias2 = tf.Variable(tf.zeros([1], dtype=tf.float32))
        
        # (w - kernel_size)/stride + 1                                          # [None, 93, 120, 1]
        h0 = tf.nn.conv2d(input, self.kernel0, [1, 1, 1, 1], 'VALID')
        h0 = h0 + self.bias0
        h0 = tf.nn.leaky_relu(h0)
        # h0 = tf.nn.avg_pool(h0, [1, 1, 2, 1], [1, 1, 1, 1], 'VALID')
        # h0 = tf.reshape(h0, [None, ])
        h0 = tf.expand_dims(tf.squeeze(h0, axis=1), axis=-1)                    # [None, 76, 32, 1]
        h1 = tf.nn.conv2d(h0, self.kernel1, [1, 1, 1, 1], 'VALID')
        h1 = h1 + self.bias1
        h1 = tf.nn.leaky_relu(h1)
        # h1 = tf.nn.avg_pool(h1, [2, 2], [1, 1])
        
        h1 = tf.expand_dims(tf.squeeze(h1, axis=1), axis=-1)                    # [None, 17, 32, 1]
        h1 = tf.nn.avg_pool(h1, [1, 1, 2, 1], [1, 1, 2, 1], 'VALID')  # [None, 17, 16, 1]
        
        h2 = tf.nn.conv2d(h1, self.kernel2, [1, 1, 1, 1], 'VALID')
        h2 = h2 + self.bias2
        h2 = tf.nn.softmax(h2)                                                  # Shape = [None, 1, 8, 1]

        self.w01 = tf.Variable(tf.random.normal([self.hdim * self.wdim, 256], stddev=0.6, dtype=tf.float32))
        self.b01 = tf.Variable(tf.zeros([256], dtype=tf.float32))
        self.w11 = tf.Variable(tf.random.normal([256, 64], stddev=0.4, dtype=tf.float32))
        self.b11 = tf.Variable(tf.zeros([64], dtype=tf.float32))
        self.w21 = tf.Variable(tf.random.normal([64, 8], stddev=0.5, dtype=tf.float32))
        self.b21 = tf.Variable(tf.zeros([8], dtype=tf.float32))
                # h0 = tf.reshape(input, [-1, self.hdim * self.wdim])
        h0 = tf.matmul(input, self.w01) + self.b01
        h0 = tf.nn.sigmoid(h0)
                h1 = tf.matmul(h0, self.w11) + self.b11
        h1 = tf.nn.sigmoid(h1)
                h2 = tf.matmul(h1, self.w21) + self.b21
        h2 = tf.nn.softmax(h2)
        '''
        self.ew0 = tf.Variable(tf.random.normal([self.hdim * self.wdim, 256], stddev=0.06, dtype=tf.float32))
        self.eb0 = tf.Variable(tf.zeros([256], dtype=tf.float32))
        self.ew1 = tf.Variable(tf.random.normal([256, 64], stddev=0.1, dtype=tf.float32))
        self.eb1 = tf.Variable(tf.zeros([64], dtype=tf.float32))
        self.ew2 = tf.Variable(tf.random.normal([64, 8], stddev=0.07, dtype=tf.float32))
        self.eb2 = tf.Variable(tf.zeros([8], dtype=tf.float32))
        self.ew3 = tf.Variable(tf.random.normal([256, 64], stddev=0.06, dtype=tf.float32))
        self.eb3 = tf.Variable(tf.zeros([64], dtype=tf.float32))
        self.ew4 = tf.Variable(tf.random.normal([64, 8], stddev=0.4, dtype=tf.float32))
        self.eb4 = tf.Variable(tf.zeros([8], dtype=tf.float32))

        h0 = tf.matmul(input, self.ew0) + self.eb0
        h0 = tf.nn.elu(h0)

        mean1 = tf.matmul(h0, self.ew1) + self.eb1
        mean1 = tf.nn.elu(mean1)

        mean2 = tf.matmul(mean1, self.ew2) + self.eb2

        logv1 = tf.matmul(h0, self.ew3) + self.eb3
        logv1 = tf.nn.elu(logv1)

        logv2 = tf.matmul(logv1, self.ew4) + self.eb4

        return mean2, logv2

    def Gaussion(self, mean, log):
        eps = tf.random_normal(shape=tf.shape(mean))
        return mean + tf.multiply(tf.exp(log * 0.5), eps)

    def build_decoder(self, feature_map):
        # self.kernel01 = tf.Variable(tf.random.normal([], stddev=0.6, dtype=tf.float32))
        # self.bias01 = tf.Variable(tf.zeros([], dtype=tf.float32))
        # self.kernel11 = tf.Variable(tf.random.normal([], stddev=0.4, dtype=tf.float32))
        # self.bias11 = tf.Variable(tf.zeros([], dtype=tf.float32))
        # self.kernel21 = tf.Variable(tf.random.normal([], stddev=0.5, dtype=tf.float32))
        # self.bias21 = tf.Variable(tf.zeros([], dtype=tf.float32))

        self.w0 = tf.Variable(tf.random.normal([8, 64], stddev=0.414, dtype=tf.float32))
        self.b0 = tf.Variable(tf.zeros([64], dtype=tf.float32))
        self.w1 = tf.Variable(tf.random.normal([64, 256], stddev=0.4, dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([256], dtype=tf.float32))
        self.w2 = tf.Variable(tf.random.normal([256, self.ydim], stddev=0.256, dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([self.ydim], dtype=tf.float32))

        # h0 = tf.expand_dims(feature_map, axis=[-1])
        # h0 = tf.nn.conv2d_transpose(feature_map, self.kernel01, [], [1, 1, 1, 1])
        h0 = tf.matmul(feature_map, self.w0) + self.b0
        h0 = tf.nn.elu(h0)

        h1 = tf.matmul(h0, self.w1) + self.b1
        h1 = tf.nn.elu(h1)

        h2 = tf.matmul(h1, self.w2) + self.b2
        # h2 = tf.squeeze(h2)

        return h2
