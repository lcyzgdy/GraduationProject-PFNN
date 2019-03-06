import tensorflow as tf
import numpy as np

dataset = np.load('./data/database.npz')
X = dataset['Xun'].astype(np.float)
Y = dataset['Yun'].astype(np.float)
P = dataset['Pun'].astype(np.float)

print(X.shape, ',', Y.shape)

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

j = 31
w = ((60*2)//10)

Xstd[w*0:w * 1] = Xstd[w*0:w * 1].mean()  # Trajectory Past Positions
Xstd[w*1:w * 2] = Xstd[w*1:w * 2].mean()  # Trajectory Future Positions
Xstd[w*2:w * 3] = Xstd[w*2:w * 3].mean()  # Trajectory Past Directions
Xstd[w*3:w * 4] = Xstd[w*3:w * 4].mean()  # Trajectory Future Directions
Xstd[w*4:w*10] = Xstd[w*4:w*10].mean()  # Trajectory Gait

# Mask Out Unused Joints in Input

joint_weights = np.array([
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)

Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w *
                                   10+j*3*1].mean() / (joint_weights * 0.1)  # Pos
Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w *
                                   10+j*3*2].mean() / (joint_weights * 0.1)  # Vel
Xstd[w*10+j*3*2:] = Xstd[w*10+j*3*2:].mean()  # Terrain

Ystd[0:2] = Ystd[0:2].mean()  # Translational Velocity
Ystd[2:3] = Ystd[2:3].mean()  # Rotational Velocity
Ystd[3:4] = Ystd[3:4].mean()  # Change in Phase
Ystd[4:8] = Ystd[4:8].mean()  # Contacts

Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean()  # Trajectory Future Positions
Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean()  # Trajectory Future Directions

Xmean.astype(np.float32).tofile('./result/Xmean.bin')
Ymean.astype(np.float32).tofile('./result/Ymean.bin')
Xstd.astype(np.float32).tofile('./result/Xstd.bin')
Ystd.astype(np.float32).tofile('./result/Ystd.bin')

# Normalize Data

X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd


# Layer


