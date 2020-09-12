import numpy as np

DATASET_PATH = 'C:\\Users\\guandeyu1\\Documents\\Code\\PFNN\\'
OUTPUT_PATH = 'C:\\Users\\guandeyu1\\Documents\\Unity\\AI4Animation\\TensorFlow\\SIGGRAPH_2017\\data'

dataset = np.load(DATASET_PATH + 'database.npz')
X = dataset['Xun'].astype(np.float32)
Y = dataset['Yun'].astype(np.float32)
P = dataset['Pun'].astype(np.float32)

P = np.expand_dims(X[..., -1], axis=1)
X = np.concatenate((X, P), axis=1)

# X.tofile(OUTPUT_PATH + '\\Input.txt', ' ')
# np.savetxt(fname=OUTPUT_PATH + '\\Input.txt', X=X)
np.savetxt(fname=OUTPUT_PATH + '\\Output.txt', X=Y)
