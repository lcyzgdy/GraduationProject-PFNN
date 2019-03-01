import tensorflow as tf
import numpy as np

dataset = np.load('./data/database.npz')
X = dataset['Xun'].astype(np.float)
Y = dataset['Yun'].astype(np.float)
P = dataset['Pun'].astype(np.float)

print(X.shape, ',', Y.shape)
