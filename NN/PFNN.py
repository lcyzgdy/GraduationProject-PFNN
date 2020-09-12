import tensorflow as tf
import numpy as np
'''
class PhaseFunctionParameters:
    def __init__(self, control_num, shape, rng, phase, name):
        self.control_num = control_num
        self.weight_shape = [control_num]
        self.weight_shape.extend(shape)
        self.bias_shape = self.weight_shape[:-1]
        self.rng = rng
        self.weight_array = tf.Variable(self.init_alpha(), name=name + 'Alpha')
        self.bias_array = tf.Variable(self.init_beta(), name=name + 'Beta')

        self.phase_1, self.w_amount, self.b_amount = self.get_index_amount(phase[:, -1])
        self.phase_0 = (self.phase_1 - 1) % self.control_num
        self.phase_2 = (self.phase_1 + 1) % self.control_num
        self.phase_3 = (self.phase_1 + 2) % self.control_num

        self.weights = self.generate_weight()
        self.bias = self.generate_bias()

    def get_index_amount(self, phase):
        nslices = self.control_num
        pscale = nslices * phase
        pamount = pscale % 1.0
        pindex_1 = tf.cast(pscale, 'int32') % nslices
        b_amount = tf.expand_dims(pamount, 1)
        w_amount = tf.expand_dims(b_amount, 1)
        return pindex_1, w_amount, b_amount

    def init_alpha(self):
        shape = self.weight_shape
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(self.rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
                           dtype=np.float32)
        return tf.convert_to_tensor(alpha, dtype=tf.float32)

    def init_beta(self):
        return tf.zeros(shape=self.bias_shape, dtype=tf.float32)

    def generate_weight(self):
        w0 = tf.nn.embedding_lookup(self.weight_array, self.phase_0)
        w1 = tf.nn.embedding_lookup(self.weight_array, self.phase_1)
        w2 = tf.nn.embedding_lookup(self.weight_array, self.phase_2)
        w3 = tf.nn.embedding_lookup(self.weight_array, self.phase_3)
        w = self.w_amount
        return cubic(w, w0, w1, w2, w3)

    def generate_bias(self):
        b0 = tf.nn.embedding_lookup(self.bias_array, self.phase_0)
        b1 = tf.nn.embedding_lookup(self.bias_array, self.phase_1)
        b2 = tf.nn.embedding_lookup(self.bias_array, self.phase_2)
        b3 = tf.nn.embedding_lookup(self.bias_array, self.phase_3)
        w = self.b_amount
        return cubic(w, b0, b1, b2, b3)


    h0 = tf.expand_dims(features, -1)
    h0 = tf.nn.dropout(h0, keep_prob)

    b0 = tf.expand_dims(p0.bias, -1)
    h1 = tf.matmul(p0.weights, h0) + b0
    h1 = tf.nn.elu(h1)
    h1 = tf.nn.dropout(h1, keep_prob)

    b1 = tf.expand_dims(p1.bias, -1)
    h2 = tf.matmul(p1.weights, h1) + b1
    h2 = tf.nn.elu(h2)
    h2 = tf.nn.dropout(h2, keep_prob)

    b2 = tf.expand_dims(p2.bias, -1)
    h3 = tf.matmul(p2.weights, h2) + b2
    h3 = tf.squeeze(h3)

    cost = tf.square(labels - h3)
    loss = tf.reduce_mean(cost)
    # test_err = loss

    return h3, loss  # , test_err

'''


class PFNN:
    def __init__(self, xdim, ydim, hdim, nslices):
        self.features = tf.placeholder(tf.float32, [None, xdim], name='features')
        self.labels = tf.placeholder(tf.float32, [None, ydim], name='labels')
        self.phase = tf.placeholder(tf.float32, [None, 1], name='phase')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.nslices = nslices

        self.xdim = xdim
        self.ydim = ydim
        self.hdim = hdim

        self.network, self.loss = self.build()

    def xavier_initializer(self, shape):
        bound = np.sqrt(6. / np.prod(shape[-2:]))
        xavier = np.asarray(np.random.uniform(low=-bound, high=bound, size=shape),
                            dtype=np.float32)
        return xavier

    def cubic(self, w, a0, a1, a2, a3):
        return a1 + w * (0.5 * a2 - 0.5 * a0) + w * w * (a0 - 2.5 * a1 + 2 * a2 - 0.5 * a3) + w * w * w * (1.5 * a1 - 1.5 * a2 + 0.5 * a3 - 0.5 * a0)

    def catmull_rom(self, w, b):
        phase_1 = tf.cast(self.nslices * self.phase[:, -1], 'int32') % self.nslices
        phase_0 = (phase_1 - 1) % self.nslices
        phase_2 = (phase_1 + 1) % self.nslices
        phase_3 = (phase_1 + 2) % self.nslices

        b_amount = tf.expand_dims((self.nslices * self.phase[:, -1]) % 1.0, 1)
        w_amount = tf.expand_dims(b_amount, 1)

        w0 = tf.nn.embedding_lookup(w, phase_0)
        w1 = tf.nn.embedding_lookup(w, phase_1)
        w2 = tf.nn.embedding_lookup(w, phase_2)
        w3 = tf.nn.embedding_lookup(w, phase_3)
        w_ = self.cubic(w_amount, w0, w1, w2, w3)

        b0 = tf.nn.embedding_lookup(b, phase_0)
        b1 = tf.nn.embedding_lookup(b, phase_1)
        b2 = tf.nn.embedding_lookup(b, phase_2)
        b3 = tf.nn.embedding_lookup(b, phase_3)
        b_ = self.cubic(b_amount, b0, b1, b2, b3)

        return w_, tf.expand_dims(b_, -1)

    def build(self):
        self.w0 = tf.Variable(self.xavier_initializer(shape=[self.nslices, self.hdim, self.xdim]), dtype=tf.float32)
        self.b0 = tf.Variable(tf.zeros(shape=[self.nslices, self.hdim], dtype=tf.float32))
        self.w1 = tf.Variable(self.xavier_initializer(shape=[self.nslices, self.hdim, self.hdim]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros(shape=[self.nslices, self.hdim], dtype=tf.float32))
        self.w2 = tf.Variable(self.xavier_initializer(shape=[self.nslices, self.ydim, self.hdim]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros(shape=[self.nslices, self.ydim], dtype=tf.float32))

        w0, b0 = self.catmull_rom(self.w0, self.b0)
        w1, b1 = self.catmull_rom(self.w1, self.b1)
        w2, b2 = self.catmull_rom(self.w2, self.b2)

        hi = tf.nn.dropout(self.features, self.keep_prob)
        hi = tf.expand_dims(hi, -1)

        h0 = tf.matmul(w0, hi) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, self.keep_prob)

        h1 = tf.matmul(w1, h0) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, self.keep_prob)

        h2 = tf.matmul(w2, h1) + b2
        h2 = tf.squeeze(h2, -1)

        cost = tf.square(self.labels - h2)
        loss = tf.reduce_mean(cost)

        return h2, loss

    def save(self, sess, result_folder):
        for i in range(50):
            pscale = self.nslices * (float(i) / 50)
            # weight
            pamount = pscale % 1.0
            # index
            pindex_1 = int(pscale) % self.nslices
            pindex_0 = (pindex_1 - 1) % self.nslices
            pindex_2 = (pindex_1 + 1) % self.nslices
            pindex_3 = (pindex_1 + 2) % self.nslices

            weights = (sess.run(self.w0), sess.run(self.w1), sess.run(self.w2))
            bias = (sess.run(self.b0), sess.run(self.b1), sess.run(self.b2))

            for j in range(len(weights)):
                a = weights[j]
                b = bias[j]
                W = self.cubic(pamount, a[pindex_0], a[pindex_1], a[pindex_2], a[pindex_3])
                B = self.cubic(pamount, b[pindex_0], b[pindex_1], b[pindex_2], b[pindex_3])

                W.tofile(result_folder + 'W%0i_%03i.bin' % (j, i))
                B.tofile(result_folder + 'b%0i_%03i.bin' % (j, i))
