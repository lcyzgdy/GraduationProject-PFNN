import tensorflow as tf
import numpy as np
import sys

# Tensorflow config
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Train config
EPOCH = 16
BATCH_SIZE = 32
X_DIM = 342
# Y_DIM = 310
H_DIM = 32
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8
KEEP_PROB = 1

RESULT_FOLDER = './result/autoencoder/'

dataset = np.load('./data/newdata.npz')
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)

xMean, xStd = X.mean(axis=0), X.std(axis=0)
yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

for i in range(xStd.size):
    if (xStd[i] == 0):
        xStd[i] = 1
for i in range(yStd.size):
    if (yStd[i] == 0):
        yStd[i] = 1

xMean.astype(np.float32).tofile('./result/autoencoder/Xmean.bin')
yMean.astype(np.float32).tofile('./result/autoencoder/Ymean.bin')
xStd.astype(np.float32).tofile('./result/autoencoder/Xstd.bin')
yStd.astype(np.float32).tofile('./result/autoencoder/Ystd.bin')

X = (X - xMean) / xStd
Y = (Y - yMean) / yStd


class Mpl:
    def __init__(self, keep_prob):
        self.w0 = tf.Variable(tf.random_normal(shape=[X_DIM, H_DIM], dtype=tf.float32, stddev=0.01))
        self.b0 = tf.Variable(tf.zeros(shape=[H_DIM], dtype=tf.float32))
        # self.w1 = tf.Variable(np.random.uniform(low=-1.0, high=1.0, size=[H_DIM, H_DIM]), dtype=tf.float32)
        # self.b1 = tf.Variable(np.zeros(shape=[H_DIM], dtype=np.float32))
        # self.w2 = tf.Variable(np.random.uniform(low=-1.0, high=1.0, size=[H_DIM, X_DIM]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random_normal(shape=[H_DIM, X_DIM], dtype=tf.float32, stddev=0.01))
        self.b2 = tf.Variable(tf.zeros(shape=[X_DIM], dtype=np.float32))

        self.features = tf.placeholder(tf.float32, shape=[None, X_DIM])
        # self.labels = tf.placeholder(tf.float32, shape=[None, Y_DIM])
        self.keep_prob = tf.placeholder(tf.float32)

        self.network, self.loss = self.build_network(self.features)

        self.learning_rate = tf.placeholder(tf.float32)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON).minimize(self.loss)

    def build_network(self, features):
        # hi = tf.expand_dims(features, -1)
        hi = features
        hi = tf.nn.dropout(hi, keep_prob=self.keep_prob)

        # b0 = tf.expand_dims(self.b0, -1)
        h0 = tf.matmul(hi, self.w0) + self.b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob=self.keep_prob)

        # b2 = tf.expand_dims(self.b2, -1)
        h2 = tf.matmul(h0, self.w2) + self.b2
        # h2 = tf.squeeze(h2)

        cost = tf.square(h2 - features)
        loss = tf.reduce_mean(cost)

        return h2, loss

    def save(self, sess):
        w0 = sess.run(self.w0)
        w0.tofile(RESULT_FOLDER + 'w0.bin')
        del w0
        # w1 = sess.run(self.w1)
        ## w1.tofile(RESULT_FOLDER + '/w1.bin')
        # del w1
        w2 = sess.run(self.w2)
        w2.tofile(RESULT_FOLDER + 'w2.bin')
        del w2
        b0 = sess.run(self.b0)
        b0.tofile(RESULT_FOLDER + 'b0.bin')
        del b0
        # b1 = sess.run(self.b1)
        # b1.tofile(RESULT_FOLDER + '/b1.bin')
        # del b1
        b2 = sess.run(self.b2)
        b2.tofile(RESULT_FOLDER + 'b2.bin')
        del b2


network = Mpl(KEEP_PROB)

# tf.summary.scalar('loss', network.loss)
saver = tf.train.Saver()

DATA_LENTH = X.shape[0]
I = np.arange(DATA_LENTH)

print('Training...')
with tf.Session(config=tf_config) as sess:
    # logger = tf.summary.FileWriter('./log/mpl_log0/', sess.graph)
    # merage = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCH):
        np.random.shuffle(I)
        ave_loss = 0
        for i in range(DATA_LENTH // BATCH_SIZE):
            index = I[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = {
                network.features: X[index],
                # network.features: np.random.uniform(size=[BATCH_SIZE, X_DIM]),
                network.learning_rate: LEARNING_RATE / ((e + 1) / 8 * 10),
                network.keep_prob: KEEP_PROB
            }
            l, _ = sess.run([network.loss, network.optimizer], feed_dict=feed_dict)
            ave_loss = (ave_loss * i + l) / (i + 1)
            print('In batch %i: loss: %f' % (i, ave_loss), end='\r')
            sys.stdout.flush()

        print('Epoch %i: Total loss: %f' % (e, ave_loss))
        save_path = saver.save(sess, './model/autoencoder/model.ckpt', global_step=e)
    network.save(sess)
print('Done!')
