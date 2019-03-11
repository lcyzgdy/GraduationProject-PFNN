import tensorflow as tf
import numpy as np

# Config
EPOCH = 10
BATCH_SIZE = 64
DATASET_PATH = "C:\\Users\\guandeyu1\\Documents\\Code\\PFNN\\"

# dataset = np.load(DATASET_PATH + 'database.npz')
# X = dataset['Xun'].astype(np.float)
# Y = dataset['Yun'].astype(np.float)
# P = dataset['Pun'].astype(np.float)

# print(X.shape, ',', Y.shape, ',', P.shape)

xMean = np.fromfile('./data/Xmean.bin', dtype=np.float32)
xStd = np.fromfile('./data/Xstd.bin', dtype=np.float32)
yMean = np.fromfile('./data/Ymean.bin', dtype=np.float32)
yStd = np.fromfile('./data/Ystd.bin', dtype=np.float32)

# print(xMean.shape)
# print(xStd.shape)
# print(yMean.shape)
# print(yStd.shape)

# Normalize Data
X = (X - xMean) / xStd
Y = (Y - yMean) / yStd

# Slice Input Data
DATA_LENGTH = xMean.shape[0]
features_batch, labels_batch =\
    tf.train.batch((X, Y), batch_size=BATCH_SIZE, num_threads=8)
# Layer

features = tf.placeholder(tf.float32, [1, 342])
labels = tf.placeholder(tf.float32, [1, 311])

W0 = tf.Variable(tf.truncated_normal([342, 512]))
b0 = tf.Variable(tf.constant(0, shape=[512]))
W1 = tf.Variable(tf.truncated_normal([512, 512]))
b1 = tf.Variable(tf.constant(0, shape=[512]))
W2 = tf.Variable(tf.truncated_normal([512, 311]))
b2 = tf.Variable(tf.constant(0, shape=[311]))

network = tf.nn.elu(
    W2 * tf.nn.elu(
        W1 * tf.nn.elu(
            W0 * features + b0) + b1) + b2)

loss = tf.reduce_min(tf.square(network - labels))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())
    for e in range(EPOCH):
        feed_dict = {features: features_batch, labels: labels_batch}
        sess.run(feed_dict=feed_dict)
        pass
    coord.request_stop()
    coord.join(threads)

tf.train.Saver.save(network)


# Incremental Learning
