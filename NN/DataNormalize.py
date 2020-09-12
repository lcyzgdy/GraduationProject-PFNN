import numpy as np


def Normalize(arr, mean, std):
    arr = arr - mean
    for i in range(std.size):
        if (std[i] == 0):
            std[i] = 1
    arr = arr / std
    return arr


def mask_for_pfnn(xMean, xStd, yMean, yStd):
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


def mask_for_mlp(xMean, xStd, yMean, yStd):
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
    yStd[3:7] = yStd[3:7].mean()  # Contacts

    yStd[7 + w * 0: 7 + w * 1] = yStd[7 + w * 0: 7 + w * 1].mean()  # Trajectory Future Positions
    yStd[7 + w * 1: 7 + w * 2] = yStd[7 + w * 1: 7 + w * 2].mean()  # Trajectory Future Directions

    yStd[7 + w * 2 + j * 3 * 0: 7 + w * 2 + j * 3 * 1] = yStd[7 + w * 2 + j * 3 * 0: 7 + w * 2 + j * 3 * 1].mean()  # Pos
    yStd[7 + w * 2 + j * 3 * 1: 7 + w * 2 + j * 3 * 2] = yStd[7 + w * 2 + j * 3 * 1: 7 + w * 2 + j * 3 * 2].mean()  # Vel
    yStd[7 + w * 2 + j * 3 * 2: 7 + w * 2 + j * 3 * 3] = yStd[7 + w * 2 + j * 3 * 2: 7 + w * 2 + j * 3 * 3].mean()  # Rot

    return xMean, xStd, yMean, yStd


def mask_for_pfnn_custom(xMean, xStd, yMean, yStd):

    j = 28
    w = ((60*2)//10)
    xStd[w * 0: w * 1] = xStd[w * 0: w * 1].mean()  # Trajectory Past Positions
    xStd[w * 1: w * 2] = xStd[w * 1: w * 2].mean()  # Trajectory Future Positions
    xStd[w * 2: w * 3] = xStd[w * 2: w * 3].mean()  # Trajectory Past Directions
    xStd[w * 3: w * 4] = xStd[w * 3: w * 4].mean()  # Trajectory Future Directions
    xStd[w * 4: w * 10] = xStd[w * 4: w * 10].mean()  # Trajectory Gait

    joint_weights = np.array([
        1,                              # Hips
        1, 1, 1, 1,
        1, 1, 1,                        # Head
        1, 1, 1, 1, 1,                  # Left Arm?
        1, 1, 1, 1, 1,                  # Right Arm?
        1, 1, 1, 1, 1,                  # Left Leg?
        1, 1, 1, 1, 1]).repeat(3)       # Right Leg?

    xStd[w * 10 + j * 3 * 0: w * 10 + j * 3 * 1] = xStd[w * 10 + j * 3 * 0: w * 10 + j * 3 * 1].mean()  # / (joint_weights * 0.1)  # Pos
    xStd[w * 10 + j * 3 * 1: w * 10 + j * 3 * 2] = xStd[w * 10 + j * 3 * 1: w * 10 + j * 3 * 2].mean()  # / (joint_weights * 0.1)  # Vel
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
