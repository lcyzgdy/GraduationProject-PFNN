import tensorflow as tf
import numpy as np
'''
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Train config
EPOCH = 10
BATCH_SIZE = 64
X_DIM = 342
Y_DIM = 310
H_DIM = 512
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8
KEEP_PROB = 0.9

RESULT_FOLDER = './result/mann/'

'''


class MANN:
    def __init__(self, xdim, ydim, nslice, hdim, gate_hdim, feature_index):
        self.features = tf.placeholder(dtype=tf.float32, shape=[None, xdim], name='Features')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, ydim], name='Labels')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='KeepBlob')
        self.hdim = hdim
        self.nslice = nslice
        self.xdim = xdim - 8
        self.ydim = ydim
        self.gate_hdim = gate_hdim
        self.gate_xdim = len(feature_index)
        self.feature_index = feature_index

        self.gate_network = self.build_gate_network()
        self.network, self.loss = self.build()

        # self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

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

        # w0 = tf.reduce_sum((tf.nn.embedding_lookup(self.w0, i) * tf.nn.embedding_lookup(self.gate_network, i) for i in range(self.nslice)), axis=0)  # ?
        # b0 = tf.reduce_sum((tf.nn.embedding_lookup(self.b0, i) * tf.nn.embedding_lookup(self.gate_network, i) for i in range(self.nslice)), axis=0)
        # w1 = tf.reduce_sum((tf.nn.embedding_lookup(self.w1, i) * tf.nn.embedding_lookup(self.gate_network, i) for i in range(self.nslice)), axis=0)
        # b1 = tf.reduce_sum((tf.nn.embedding_lookup(self.b1, i) * tf.nn.embedding_lookup(self.gate_network, i) for i in range(self.nslice)), axis=0)
        # w2 = tf.reduce_sum((tf.nn.embedding_lookup(self.w2, i) * tf.nn.embedding_lookup(self.gate_network, i) for i in range(self.nslice)), axis=0)
        # b2 = tf.reduce_sum((tf.nn.embedding_lookup(self.b2, i) * tf.nn.embedding_lookup(self.gate_network, i) for i in range(self.nslice)), axis=0)
        '''
        def cond(i, w, a):
            return tf.less(i, self.nslice)

        def body(i, w, a):
            w = w + tf.nn.embedding_lookup(a, i)
            i = i + 1
            return i, w, a
        w0 = tf.Variable(np.zeros(shape=[self.xdim, self.hdim], dtype=np.float32))
        b0 = tf.Variable(np.zeros(shape=[self.hdim], dtype=np.float32))
        w1 = tf.Variable(np.zeros(shape=[self.hdim, self.hdim], dtype=np.float32))
        b1 = tf.Variable(np.zeros(shape=[self.hdim], dtype=np.float32))
        w2 = tf.Variable(np.zeros(shape=[self.hdim, self.ydim], dtype=np.float32))
        b2 = tf.Variable(np.zeros(shape=[self.ydim], dtype=np.float32))

        i = tf.constant(0)
        i, w0, _ = tf.while_loop(cond, body, [i, w0, self.w0])

        i = tf.constant(0)
        i, w1, _ = tf.while_loop(cond, body, [i, w1, self.w1])

        i = tf.constant(0)
        i, w2, _ = tf.while_loop(cond, body, [i, w2, self.w2])

        i = tf.constant(0)
        i, b0, _ = tf.while_loop(cond, body, [i, b0, self.b0])

        i = tf.constant(0)
        i, b1, _ = tf.while_loop(cond, body, [i, b1, self.b1])

        i = tf.constant(0)
        i, b2, _ = tf.while_loop(cond, body, [i, b2, self.b2])
        '''
        hi = self.features[:, :-8]
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

        # hi = tf.transpose(self.features)
        # hi = tf.nn.dropout(self.features, keep_prob=self.keep_prob)
        feature_index = self.feature_index
        hi = self.features[..., feature_index[0]:feature_index[0] + 1]
        feature_index.remove(feature_index[0])
        for i in feature_index:
            hi = tf.concat([hi, self.features[..., i:i + 1]], axis=-1)

        h0 = tf.matmul(hi, self.gw0) + self.gb0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob=self.keep_prob)

        h1 = tf.matmul(h0, self.gw1) + self.gb1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

        h2 = tf.matmul(h1, self.gw2) + self.gb2
        h2 = tf.nn.softmax(h2)

        # return tf.transpose(h2)
        return h2


'''
dataset = np.load('./data/newdata.npz')
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)

print(X.shape, Y.shape)

xMean, xStd = X.mean(axis=0), X.std(axis=0)
yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

for i in range(xStd.size):
    if (xStd[i] == 0):
        xStd[i] = 1
for i in range(yStd.size):
    if (yStd[i] == 0):
        yStd[i] = 1


xMean.astype(np.float32).tofile('./result/mann/Xmean.bin')
yMean.astype(np.float32).tofile('./result/mann/Ymean.bin')
xStd.astype(np.float32).tofile('./result/mann/Xstd.bin')
yStd.astype(np.float32).tofile('./result/mann/Ystd.bin')

X = (X - xMean) / xStd
Y = (Y - yMean) / yStd

DATA_LENGTH = X.shape[0]
I = np.arange(DATA_LENGTH)

mann = MANN(xdim=X_DIM, ydim=Y_DIM, nslice=4,  hdim=H_DIM, gate_hdim=32, keep_prob=KEEP_PROB, batch_size=BATCH_SIZE)

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCH):
        np.random.shuffle(I)
        ave_loss = 0
        for i in range(DATA_LENGTH // BATCH_SIZE):
            index = I[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = {
                mann.features: X[index],
                mann.labels: Y[index]
            }
            loss, _ = sess.run([mann.loss, mann.optimizer], feed_dict=feed_dict)
            ave_loss = (ave_loss * i + loss) / (i + 1)
            if (i % 100):
                print("In batch %d, Loss is %f" % (i, loss), end='\r')
        print("Epoch %d, Total loss is %f" % (e, ave_loss))

    mann.save(sess)
'''
