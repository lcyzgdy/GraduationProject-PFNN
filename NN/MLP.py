import tensorflow as tf
import numpy as np
'''
import sys

# Tensorflow config

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Train config

EPOCH = 10
BATCH_SIZE = 4
X_DIM = 342
Y_DIM = 310
H_DIM = 512
LEARNING_RATE = 0.00001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8
KEEP_PROB = 0.7

RESULT_FOLDER = './result/Mlp/'

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

xMean.astype(np.float32).tofile('./result/Mlp/Xmean.bin')
yMean.astype(np.float32).tofile('./result/Mlp/Ymean.bin')
xStd.astype(np.float32).tofile('./result/Mlp/Xstd.bin')
yStd.astype(np.float32).tofile('./result/Mlp/Ystd.bin')

# X.astype(np.float32).tofile('./result/Mlp/X.bin')
# Y.astype(np.float32).tofile('./result/Mlp/Y.bin')
# exit()
X = (X - xMean) / xStd
Y = (Y - yMean) / yStd
'''


class Mlp:
    def __init__(self, xdim, ydim, hdim):
        self.w0 = tf.Variable(tf.random_normal(shape=[xdim, hdim], dtype=tf.float32, stddev=0.051))
        self.b0 = tf.Variable(tf.zeros(shape=[hdim], dtype=tf.float32))
        self.w1 = tf.Variable(tf.random_normal(shape=[hdim, hdim], dtype=tf.float32, stddev=0.051))
        self.b1 = tf.Variable(tf.zeros(shape=[hdim], dtype=tf.float32))
        self.w2 = tf.Variable(tf.random_normal(shape=[hdim, ydim], dtype=tf.float32, stddev=0.051))
        self.b2 = tf.Variable(tf.zeros(shape=[ydim], dtype=tf.float32))

        self.features = tf.placeholder(tf.float32, shape=[None, xdim])
        self.labels = tf.placeholder(tf.float32, shape=[None, ydim])
        self.keep_prob = tf.placeholder(tf.float32)

        self.network, self.loss = self.build_network(self.features, self.labels)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON).minimize(self.loss)

    def build_network(self, features, labels):
        # hi = tf.expand_dims(features, -1)
        hi = features
        hi = tf.nn.dropout(hi, keep_prob=self.keep_prob)

        # b0 = tf.expand_dims(self.b0, -1)
        h0 = tf.matmul(hi, self.w0)
        h0 = h0 + self.b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob=self.keep_prob)

        # b1 = tf.expand_dims(self.b1, -1)
        h1 = tf.matmul(h0, self.w1) + self.b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

        # b2 = tf.expand_dims(self.b2, -1)
        h2 = tf.matmul(h1, self.w2) + self.b2
        # h2 = tf.squeeze(h2)

        cost = tf.square(h2 - labels)
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


'''
network = Mlp(KEEP_PROB)

# tf.summary.scalar('loss', network.loss)
saver = tf.train.Saver()

DATA_LENTH = X.shape[0]
I = np.arange(DATA_LENTH)

print('Training...')
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCH):
        np.random.shuffle(I)
        ave_loss = 0
        for i in range(DATA_LENTH // BATCH_SIZE):
            index = I[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = {
                network.features: X[index],
                network.labels: Y[index],
                network.keep_prob: KEEP_PROB
            }
            l, _ = sess.run([network.loss, network.optimizer], feed_dict=feed_dict)
            ave_loss = (ave_loss * i + l) / (i + 1)
            print('In batch %i: loss: %f' % (i, ave_loss), end='\r')
            sys.stdout.flush()

        print('Epoch %i: Total loss: %f' % (e, ave_loss))
        save_path = saver.save(sess, './model/Mlp/model.ckpt', global_step=e)
    network.save(sess)
print('Done!')
'''
