import tensorflow as tf
import numpy as np
import sys

from DataNormalize import *
from MLP import Mlp
from PFNN import PFNN
from MANN import MANN
from Sin import SinNN
from VariationalEncoderDecoder import VariationalEncoderDecoder
from ConvolutionalEncoderDecoder import ConvolutionalEncoderDecoder
from MannWithPhase import MannWithPhase

# Tensorflow config
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Train config
EPOCH = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8
KEEP_PROB = 0.7

TEST_BATCH_COUNT = 100
# -------  MLP --------------------------------------------------------------------
'''
X_DIM = 342
Y_DIM = 310
H_DIM = 512

DATASET_PATH = "C:\\Users\\PC\\Documents\\Code\\PFNN\\"
RESULT_PATH = './result/mlp/'
DATASET_NAME = 'database.npz'

dataset = np.load(DATASET_PATH + DATASET_NAME)
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)

xMean, xStd = X.mean(axis=0), X.std(axis=0)
yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

xMean, xStd, yMean, yStd = mask_for_mlp(xMean, xStd, yMean, yStd)

xMean.astype(np.float32).tofile('./result/mlp/Xmean.bin')
yMean.astype(np.float32).tofile('./result/mlp/Ymean.bin')
xStd.astype(np.float32).tofile('./result/mlp/Xstd.bin')
yStd.astype(np.float32).tofile('./result/mlp/Ystd.bin')

X = Normalize(X, xMean, xStd)
Y = Normalize(Y, yMean, yStd)

DATA_LENGTH = X.shape[0]

nn = Mlp(X_DIM, Y_DIM, H_DIM)
'''
# ------------------------------------------------------------------------------------
# ----- VAE --------------------------------------------------------------------------
'''
SAMPLE_WINDOW = 120
DATASET_PATH = './data/animations'
RESULT_PATH = './result/vae/'

X = np.fromfile('%s/%s' % (DATASET_PATH, 'features.bin'), dtype=np.float32)
Y = np.fromfile('%s/%s' % (DATASET_PATH, 'labels.bin'), dtype=np.float32)
X = np.reshape(X, [X.shape[0] // (31 * 3 * SAMPLE_WINDOW), 31 * SAMPLE_WINDOW * 3])
Y = np.reshape(Y, [Y.shape[0] // (31 * 3), 31 * 3])

print(X.shape, Y.shape)

DATA_LENGTH = X.shape[0]

nn = VariationalEncoderDecoder(93, SAMPLE_WINDOW, 93)
'''
# --------------------------------------------------------------------------
# ------- Convolutional ----------------------------------------------------
'''
SAMPLE_WINDOW = 120
DATASET_PATH = './data/animations'
RESULT_PATH = './result/conv/'

X = np.fromfile('%s/%s' % (DATASET_PATH, 'features.bin'), dtype=np.float32)
Y = np.fromfile('%s/%s' % (DATASET_PATH, 'labels.bin'), dtype=np.float32)
print(X.shape, Y.shape)
X = np.reshape(X, [X.shape[0] // (31 * 3 * SAMPLE_WINDOW), 31 * 3, SAMPLE_WINDOW, 1])
Y = np.reshape(Y, [Y.shape[0] // (31 * 3), 31 * 3, 1, 1])
print(X.shape, Y.shape)

xMean = np.fromfile('./data/Xmean.bin', dtype=np.float32)
xStd = np.fromfile('./data/Xstd.bin', dtype=np.float32)
yMean = np.fromfile('./data/Ymean.bin', dtype=np.float32)
yStd = np.fromfile('./data/Ystd.bin', dtype=np.float32)

DATA_LENGTH = X.shape[0]

nn = ConvolutionalEncoderDecoder(31 * 3, SAMPLE_WINDOW, 31 * 3)
'''
# ----------------------------------------------------------------------------------
# ------- MANN -------------------------------------------------------------------------------
'''
X_DIM = 342
Y_DIM = 318
H_DIM = 512
RESULT_PATH = './result/mann/'
DATASET_NAME = 'database.npz'
DATASET_PATH = './data/animations'

dataset = np.load('%s/%s' % (DATASET_PATH, DATASET_NAME))
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)
F = dataset['Fun'].astype(np.float32)
DATA_LENGTH = X.shape[0]

X = Normalize(X, np.fromfile('./data/Xmean.bin', dtype=np.float32), np.fromfile('./data/Xstd.bin', dtype=np.float32))
Y = Normalize(Y, np.fromfile('./data/Ymean.bin', dtype=np.float32), np.fromfile('./data/Ystd.bin', dtype=np.float32))
X = np.concatenate((X, F), axis=-1)
print(X.shape, Y.shape)

nn = MANN(xdim=X_DIM, ydim=Y_DIM, nslice=4, hdim=H_DIM, gate_hdim=32, feature_index=[342, 343, 344, 345, 346, 347, 348, 349,    # Feature
                                                                                     84, 85, 86, 87, 88, 89])                   # Gait
'''
# --------------------------------------------------------------------------------------------
# ---------- PFNN Standard----------------------------------------------------------

X_DIM = 342
Y_DIM = 311
H_DIM = 512
# X_DIM = 342 + 8
# Y_DIM = 310 + 8

DATASET_PATH = "C:/Users/PC/Documents/Code/PFNN/"
RESULT_PATH = './result/pfnn/'
DATASET_NAME = 'database.npz'

dataset = np.load(DATASET_PATH + DATASET_NAME)
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)
P = dataset['Pun'].astype(np.float32)
P = np.expand_dims(P, axis=1)

xMean, xStd = X.mean(axis=0), X.std(axis=0)
yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

xMean, xStd, yMean, yStd = mask_for_pfnn(xMean, xStd, yMean, yStd)

xMean.astype(np.float32).tofile(RESULT_PATH + 'Xmean.bin')
yMean.astype(np.float32).tofile(RESULT_PATH + 'Ymean.bin')
xStd.astype(np.float32).tofile(RESULT_PATH + 'Xstd.bin')
yStd.astype(np.float32).tofile(RESULT_PATH + 'Ystd.bin')

X = Normalize(X, xMean, xStd)
Y = Normalize(Y, yMean, yStd)

DATA_LENGTH = X.shape[0]

nn = PFNN(X_DIM, Y_DIM, H_DIM, 4)

# ------------------------------------------------------------------------------
# ---------- PFNN Custom -------------------------------------------------------
'''
X_DIM = 324
Y_DIM = 284
H_DIM = 512

DATASET_PATH = "./data/"
RESULT_PATH = './result/pfnn_custom/'
DATASET_NAME = 'mydatabase.npz'

dataset = np.load(DATASET_PATH + DATASET_NAME)
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)
P = dataset['Pun'].astype(np.float32)
P = np.expand_dims(P, axis=1)

xMean, xStd = X.mean(axis=0), X.std(axis=0)
yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

xMean, xStd, yMean, yStd = mask_for_pfnn_custom(xMean, xStd, yMean, yStd)

xMean.astype(np.float32).tofile(RESULT_PATH + 'Xmean.bin')
yMean.astype(np.float32).tofile(RESULT_PATH + 'Ymean.bin')
xStd.astype(np.float32).tofile(RESULT_PATH + 'Xstd.bin')
yStd.astype(np.float32).tofile(RESULT_PATH + 'Ystd.bin')

X = Normalize(X, xMean, xStd)
Y = Normalize(Y, yMean, yStd)

DATA_LENGTH = X.shape[0]

nn = PFNN(X_DIM, Y_DIM, H_DIM, 4)
'''
# ------------------------------------------------------------------------------
# ------- MANN With Phase ------------------------------------------------------
'''
X_DIM = 342
Y_DIM = 311
H_DIM = 512

DATASET_PATH = "E:/Dataset/"
RESULT_PATH = './result/mann_with_phase2/'
DATASET_NAME = 'database.npz'

dataset = np.load(DATASET_PATH + DATASET_NAME)
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)
P = dataset['Pun'].astype(np.float32)
P = np.expand_dims(P, axis=1)
print(X.shape, Y.shape)
# xMean, xStd = X.mean(axis=0), X.std(axis=0)
# yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

# xMean, xStd, yMean, yStd = mask_for_pfnn(xMean, xStd, yMean, yStd)

# xMean.astype(np.float32).tofile(RESULT_PATH + 'Xmean.bin')
# yMean.astype(np.float32).tofile(RESULT_PATH + 'Ymean.bin')
# xStd.astype(np.float32).tofile(RESULT_PATH + 'Xstd.bin')
# yStd.astype(np.float32).tofile(RESULT_PATH + 'Ystd.bin')
xMean = np.fromfile(RESULT_PATH + 'Xmean.bin', np.float32)
yMean = np.fromfile(RESULT_PATH + 'Ymean.bin', np.float32)
xStd = np.fromfile(RESULT_PATH + 'Xstd.bin', np.float32)
yStd = np.fromfile(RESULT_PATH + 'Ystd.bin', np.float32)

X = Normalize(X, xMean, xStd)
Y = Normalize(Y, yMean, yStd)

DATA_LENGTH = X.shape[0]

nn = MannWithPhase(xdim=X_DIM, ydim=Y_DIM, nslice=4, hdim=H_DIM, gate_hdim=64)
'''
# ------------------------------------------------------------------------------

optimizer_adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON).minimize(nn.loss)
optimizer_sgd = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(nn.loss)

I = np.arange(DATA_LENGTH)

saver = tf.train.Saver(max_to_keep=5)

if __name__ == '__main__':
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        # f = open("pfnn.csv", "w")
        # flag = False
        for e in range(EPOCH):
            np.random.shuffle(I)
            ave_train_loss = 0
            ave_test_loss = 0
            optimizer = optimizer_adam if (e / EPOCH) < 0.75 else optimizer_sgd
            for i in range(DATA_LENGTH // BATCH_SIZE - TEST_BATCH_COUNT):
                index = I[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                feed_dict = {
                    nn.keep_prob: KEEP_PROB,
                    nn.features: X[index],
                    nn.labels: Y[index],
                    nn.phase: P[index]
                }
                l, _ = sess.run([nn.loss,  optimizer], feed_dict=feed_dict)
                ave_train_loss = (ave_train_loss * i + l) / (i + 1)
                if i % 2 == 0:
                    print('In batch %i: loss: %f' % (i, ave_train_loss), end='\r')
                    # if(ave_train_loss < 10 or flag):
                    #     f.write("%f\n" % ave_train_loss)
                    #     flag = True
                    sys.stdout.flush()

            for i in range(TEST_BATCH_COUNT):
                if i == 0:
                    index = I[-(i + 1) * BATCH_SIZE:]
                else:
                    index = I[-(i + 1) * BATCH_SIZE: -i * BATCH_SIZE]
                feed_dict = {
                    nn.keep_prob: 1,
                    nn.features: X[index],
                    nn.labels: Y[index],
                    nn.phase: P[index]
                }
                l = sess.run(nn.loss, feed_dict=feed_dict)
                ave_test_loss += l / TEST_BATCH_COUNT

            print('Epoch %i: Train loss: %f, Test loss: %f' % (e, ave_train_loss, ave_test_loss))
            # save_path = saver.save(sess, './model/mann_with_phase/model.ckpt', global_step=e)
            nn.save(sess, RESULT_PATH)
        # f.close()

print('Done!')
