import tensorflow as tf
import numpy as np
import sys
from CatmullRom import *

# Tensorflow config
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Config
EPOCH = 20
BATCH_SIZE = 32
DATASET_PATH = "./data/myown/test/"
LEARNING_RATE = 0.0001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8
KEEP_PROB = 0.7
X_DIM = 342
Y_DIM = 311
H_DIM = 512
TEST_BATCH = 100

dataset = np.load(DATASET_PATH + 'database.npz')
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)
P = dataset['Pun'].astype(np.float32)
P = np.expand_dims(P, axis=1)
print([X.shape, Y.shape, P.shape])


def mask(xMean, xStd, yMean, yStd):
    j = 31
    w = ((60*2)//10)
    xStd[w * 0: w * 1] = xStd[w * 0: w * 1].mean()  # Trajectory Past Positions
    xStd[w * 1: w * 2] = xStd[w * 1: w * 2].mean()  # Trajectory Future Positions
    xStd[w * 2: w * 3] = xStd[w * 2: w * 3].mean()  # Trajectory Past Directions
    xStd[w * 3: w * 4] = xStd[w * 3: w * 4].mean()  # Trajectory Future Directions
    xStd[w * 4: w * 10] = xStd[w * 4: w * 10].mean()  # Trajectory Gait

    joint_weights = np.array([
        1,
        1e-10, 1, 1, 1, 1,
        1e-10, 1, 1, 1, 1,
        1e-10, 1, 1,
        1e-10, 1, 1,
        1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
        1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

    xStd[w * 10 + j * 3 * 0: w * 10 + j * 3 * 1] = xStd[w * 10 + j * 3 * 0: w * 10 + j * 3 * 1].mean() / (joint_weights * 0.1)  # Pos
    xStd[w * 10 + j * 3 * 1: w * 10 + j * 3 * 2] = xStd[w * 10 + j * 3 * 1: w * 10 + j * 3 * 2].mean() / (joint_weights * 0.1)  # Vel
    xStd[w * 10 + j * 3 * 2:] = xStd[w * 10 + j * 3 * 2:].mean()  # Terrain

    yStd[0:2] = yStd[0:2].mean()  # Translational Velocity
    yStd[2:3] = yStd[2:3].mean()  # Rotational Velocity
    yStd[3:4] = yStd[3:4].mean()  # Change in Phase
    yStd[4:8] = yStd[4:8].mean()  # Contacts

    yStd[8 + w * 0: 8 + w * 1] = yStd[8 + w * 0: 8 + w * 1].mean()  # Trajectory Future Positions
    yStd[8 + w * 1: 8 + w * 2] = yStd[8 + w * 1: 8 + w * 2].mean()  # Trajectory Future Directions

    yStd[8 + w * 2 + j * 3 * 0: 8 + w * 2 + j * 3 * 1] = yStd[8 + w * 2 + j * 3 * 0: 8 + w * 2 + j * 3 * 1].mean()  # Pos
    yStd[8 + w * 2 + j * 3 * 1: 8 + w * 2 + j * 3 * 2] = yStd[8 + w * 2 + j * 3 * 1: 8 + w * 2 + j * 3 * 2].mean()  # Vel
    yStd[8 + w * 2 + j * 3 * 2: 8 + w * 2 + j * 3 * 3] = yStd[8 + w * 2 + j * 3 * 2: 8 + w * 2 + j * 3 * 3].mean()  # Rot

    return xMean, xStd, yMean, yStd


xMean, xStd = X.mean(axis=0), X.std(axis=0)
yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

xMean, xStd, yMean, yStd = mask(xMean=xMean, xStd=xStd, yMean=yMean, yStd=yStd)
'''
for i in range(xStd.size):
    if (xStd[i] == 0):
        xStd[i] = 1
for i in range(yStd.size):
    if (yStd[i] == 0):
        yStd[i] = 1
'''

xMean.astype(np.float32).tofile('./result/pfnn/Xmean.bin')
yMean.astype(np.float32).tofile('./result/pfnn/Ymean.bin')
xStd.astype(np.float32).tofile('./result/pfnn/Xstd.bin')
yStd.astype(np.float32).tofile('./result/pfnn/Ystd.bin')

# 归一化
X = (X - xMean)
X = X / xStd
Y = (Y - yMean)
Y = Y / yStd


class PhaseFunctionParameters:
    def __init__(self, control_num, shape, rng, phase, name):
        self.control_num = control_num
        self.weight_shape = [control_num]
        self.weight_shape.extend(shape)
        self.bias_shape = self.weight_shape[:-1]
        self.rng = rng
        self.weight_array = tf.Variable(self.xavier_initializer(), name=name + 'Weight')
        self.bias_array = tf.Variable(tf.zeros(shape=self.bias_shape, dtype=tf.float32), name=name + 'Bias')

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

    def xavier_initializer(self):
        shape = self.weight_shape
        bound = np.sqrt(6. / np.prod(shape[-2:]))
        xavier = np.asarray(self.rng.uniform(low=-bound, high=bound, size=shape),
                           dtype=np.float32)
        return xavier

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


def build_network(features, labels, p0, p1, p2, keep_prob):
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


def save_network(alpha, beta):
    nslices = 4
    for i in range(50):
        """calculate the index and weights in phase function """
        pscale = nslices * (float(i) / 50)
        # weight
        pamount = pscale % 1.0
        # index
        pindex_1 = int(pscale) % nslices
        pindex_0 = (pindex_1-1) % nslices
        pindex_2 = (pindex_1+1) % nslices
        pindex_3 = (pindex_1+2) % nslices

        for j in range(len(alpha)):
            a = alpha[j]
            b = beta[j]
            W = cubic(pamount, a[pindex_0], a[pindex_1], a[pindex_2], a[pindex_3])
            B = cubic(pamount, b[pindex_0], b[pindex_1], b[pindex_2], b[pindex_3])

            W.tofile('./result/pfnn/W%0i_%03i.bin' % (j, i))
            B.tofile('./result/pfnn/b%0i_%03i.bin' % (j, i))


features = tf.placeholder(tf.float32, [None, X_DIM], name='features')
labels = tf.placeholder(tf.float32, [None, Y_DIM], name='labels')
phase = tf.placeholder(tf.float32, [None, 1], name='phase')

rng = np.random.RandomState(23456)
nslice = 4
p0 = PhaseFunctionParameters(nslice, [H_DIM, X_DIM], rng, phase, name='p0')
p1 = PhaseFunctionParameters(nslice, [H_DIM, H_DIM], rng, phase, name='p1')
p2 = PhaseFunctionParameters(nslice, [Y_DIM, H_DIM], rng, phase, name='p3')
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
network, loss = build_network(features=features, labels=labels,
                              p0=p0, p1=p1, p2=p2, keep_prob=keep_prob)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON).minimize(loss)

# Slice data
DATA_LENGTH = X.shape[0]

I = np.arange(DATA_LENGTH)
saver = tf.train.Saver(max_to_keep=50)

print('Start trainging...')
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(EPOCH):
        ave_loss_train = 0
        ave_loss_test = 0

        np.random.shuffle(I)
        for i in range(DATA_LENGTH // BATCH_SIZE - TEST_BATCH):
            index_train = I[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = {
                features: X[index_train],
                labels: Y[index_train],
                phase: P[index_train],
                keep_prob: KEEP_PROB,
                learning_rate: LEARNING_RATE / (float(e // 5 + 1))
            }
            l, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            ave_loss_train = (ave_loss_train * i + l) / (i + 1)
            print('In batch %i: loss: %f' % (i, ave_loss_train), end='\r')
            sys.stdout.flush()

        for i in range(TEST_BATCH):
            if(i == 0):
                index_test = I[-(i + 1) * BATCH_SIZE:]
            else:
                index_test = I[-(i + 1) * BATCH_SIZE: -i * BATCH_SIZE]
            feed_dict = {
                features: X[index_test],
                labels: Y[index_test],
                phase: P[index_test],
                keep_prob: 1
            }
            l = sess.run(loss, feed_dict=feed_dict)
            ave_loss_test += l / TEST_BATCH

        print('Epoch %i: Total loss: %f, Test loss: %f' % (e, ave_loss_train, ave_loss_test))
        save_path = saver.save(sess, './model/pfnn/model.ckpt', global_step=e)

    save_network((sess.run(p0.weight_array), sess.run(p1.weight_array), sess.run(p2.weight_array)),
                 (sess.run(p0.bias_array), sess.run(p1.bias_array), sess.run(p2.bias_array)))

print('Done!')
