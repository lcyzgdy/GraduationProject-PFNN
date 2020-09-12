import sys
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

sys.path.append('./data/motion')
sys.path.append('${Workspace}')

from Learning import RBF
from Pivots import Pivots
from Quaternions import Quaternions
import Animation as Animation
import BVH as BVH

""" Options """

rng = np.random.RandomState(1234)
to_meters = 1
window = 60
njoints = 28

""" Data """

data_terrain = [
    './data/myown/Flat8.bvh',
    './data/myown/Flat4.bvh',
    # './data/myown/Flat1.bvh',
    './data/myown/Flat0.bvh',
    './data/myown/WalkingUpSteps01.bvh',
    './data/myown/WalkingUpSteps02.bvh',

    './data/myown/Flat8_mirror.bvh',
    './data/myown/Flat4_mirror.bvh',
    # './data/myown/Flat1_mirror.bvh',
    './data/myown/Flat0_mirror.bvh',
    './data/myown/WalkingUpSteps01_mirror.bvh',
    './data/myown/WalkingUpSteps02_mirror.bvh'
]

""" Load Terrain Patches """

patches_database = np.load('./data/patches.npz')
patches = patches_database['X'].astype(np.float32)
patches_coord = patches_database['C'].astype(np.float32)

""" Processing Functions """


def process_data(anim, phase, gait, type='flat'):
    """ Do FK """
    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
    global_rotations = Quaternions.from_transforms(global_xforms)

    """ Extract Forward Direction """

    sdr_l, sdr_r, hip_l, hip_r = 14, 9, 24, 19
    across = (
        (global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
        (global_positions[:, hip_l] - global_positions[:, hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    """ Smooth Forward Direction """

    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    root_rotation = Quaternions.between(forward,
                                        np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]

    """ Local Space """

    local_positions = global_positions.copy()
    local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
    local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]

    local_positions = root_rotation[:-1] * local_positions[:-1]
    local_velocities = root_rotation[:-1] * (global_positions[1:] - global_positions[:-1])
    local_rotations = abs((root_rotation[:-1] * global_rotations[:-1])).log()

    root_velocity = root_rotation[:-1] * (global_positions[1:, 0:1] - global_positions[:-1, 0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps

    """ Foot Contacts """

    fid_l, fid_r = np.array([26, 27]), np.array([21, 22])
    velfactor = np.array([0.02, 0.02])

    feet_l_x = (global_positions[1:, fid_l, 0] - global_positions[:-1, fid_l, 0])**2
    feet_l_y = (global_positions[1:, fid_l, 1] - global_positions[:-1, fid_l, 1])**2
    feet_l_z = (global_positions[1:, fid_l, 2] - global_positions[:-1, fid_l, 2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)

    feet_r_x = (global_positions[1:, fid_r, 0] - global_positions[:-1, fid_r, 0])**2
    feet_r_y = (global_positions[1:, fid_r, 1] - global_positions[:-1, fid_r, 1])**2
    feet_r_z = (global_positions[1:, fid_r, 2] - global_positions[:-1, fid_r, 2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)

    """ Phase """

    dphase = phase[1:] - phase[:-1]
    dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]

    """ Adjust Crouching Gait Value """

    if type == 'flat':
        pass
        # crouch_low, crouch_high = 80, 130
        # head = 7
        # gait[:-1, 3] = 1 - np.clip((global_positions[:-1, head, 1] - 80) / (130 - 80), 0, 1)
        # gait[-1, 3] = gait[-2, 3]

    """ Start Windows """

    Pc, Xc, Yc = [], [], []

    for i in range(window, len(anim)-window-1, 1):

        rootposs = root_rotation[i:i+1, 0] * (global_positions[i-window:i+window:10, 0] - global_positions[i:i+1, 0])
        rootdirs = root_rotation[i:i+1, 0] * forward[i-window:i+window:10]
        rootgait = gait[i-window:i+window:10]

        Pc.append(phase[i])

        Xc.append(np.hstack([
            rootposs[:, 0].ravel(), rootposs[:, 2].ravel(),  # Trajectory Pos
            rootdirs[:, 0].ravel(), rootdirs[:, 2].ravel(),  # Trajectory Dir
            rootgait[:, 0].ravel(), rootgait[:, 1].ravel(),  # Trajectory Gait
            rootgait[:, 2].ravel(), rootgait[:, 3].ravel(),
            rootgait[:, 4].ravel(), rootgait[:, 5].ravel(),
            local_positions[i-1].ravel(),  # Joint Pos
            local_velocities[i-1].ravel(),  # Joint Vel
        ]))

        rootposs_next = root_rotation[i+1:i+2, 0] * (global_positions[i+1:i+window+1:10, 0] - global_positions[i+1:i+2, 0])
        rootdirs_next = root_rotation[i+1:i+2, 0] * forward[i+1:i+window+1:10]

        Yc.append(np.hstack([
            root_velocity[i, 0, 0].ravel(),  # Root Vel X
            root_velocity[i, 0, 2].ravel(),  # Root Vel Y
            root_rvelocity[i].ravel(),    # Root Rot Vel
            dphase[i],                    # Change in Phase
            np.concatenate([feet_l[i], feet_r[i]], axis=-1),  # Contacts
            rootposs_next[:, 0].ravel(), rootposs_next[:,
                                                       2].ravel(),  # Next Trajectory Pos
            rootdirs_next[:, 0].ravel(), rootdirs_next[:,
                                                       2].ravel(),  # Next Trajectory Dir
            local_positions[i].ravel(),   # Joint Pos
            local_velocities[i].ravel(),  # Joint Vel
            local_rotations[i].ravel()    # Joint Rot
        ]))

    return np.array(Pc), np.array(Xc), np.array(Yc)


""" Sampling Patch Heightmap """


def patchfunc(P, Xp, hscale=3.937007874, vscale=3.0):

    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])

    A = np.fmod(Xp, 1.0)
    X0 = np.clip(np.floor(Xp).astype(np.int), 0,
                 np.array([P.shape[1]-1, P.shape[2]-1]))
    X1 = np.clip(np.ceil(Xp).astype(np.int), 0,
                 np.array([P.shape[1]-1, P.shape[2]-1]))

    H0 = P[:, X0[:, 0], X0[:, 1]]
    H1 = P[:, X0[:, 0], X1[:, 1]]
    H2 = P[:, X1[:, 0], X0[:, 1]]
    H3 = P[:, X1[:, 0], X1[:, 1]]

    HL = (1-A[:, 0]) * H0 + (A[:, 0]) * H2
    HR = (1-A[:, 0]) * H1 + (A[:, 0]) * H3

    return (vscale * ((1-A[:, 1]) * HL + (A[:, 1]) * HR))[..., np.newaxis]


def process_heights(anim, nsamples=10, type='flat'):
    """ Do FK """

    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
    global_rotations = Quaternions.from_transforms(global_xforms)

    """ Extract Forward Direction """

    sdr_l, sdr_r, hip_l, hip_r = 14, 9, 24, 19
    across = (
        (global_positions[:, sdr_l] - global_positions[:, sdr_r]) +
        (global_positions[:, hip_l] - global_positions[:, hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    """ Smooth Forward Direction """

    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0, 1, 0]])), direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    root_rotation = Quaternions.between(forward,
                                        np.array([[0, 0, 1]]).repeat(len(forward), axis=0))[:, np.newaxis]

    """ Foot Contacts """

    fid_l, fid_r = np.array([26, 27]), np.array([21, 22])
    velfactor = np.array([0.02, 0.02])

    feet_l_x = (global_positions[1:, fid_l, 0] -
                global_positions[:-1, fid_l, 0])**2
    feet_l_y = (global_positions[1:, fid_l, 1] -
                global_positions[:-1, fid_l, 1])**2
    feet_l_z = (global_positions[1:, fid_l, 2] -
                global_positions[:-1, fid_l, 2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor))

    feet_r_x = (global_positions[1:, fid_r, 0] -
                global_positions[:-1, fid_r, 0])**2
    feet_r_y = (global_positions[1:, fid_r, 1] -
                global_positions[:-1, fid_r, 1])**2
    feet_r_z = (global_positions[1:, fid_r, 2] -
                global_positions[:-1, fid_r, 2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor))

    feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)

    """ Toe and Heel Heights """

    toe_h, heel_h = 4.0, 5.0
    # toe_h, heel_h = 1.0, 2.0
    toe_h, heel_h = 20.0, 25.0

    """ Foot Down Positions """

    feet_down = np.concatenate([
        global_positions[feet_l[:, 0], fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_l[:, 1], fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[feet_r[:, 0], fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_r[:, 1], fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)

    """ Foot Up Positions """

    feet_up = np.concatenate([
        global_positions[~feet_l[:, 0], fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_l[:, 1], fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[~feet_r[:, 0], fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_r[:, 1], fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)

    """ Down Locations """

    feet_down_xz = np.concatenate([feet_down[:, 0:1], feet_down[:, 2:3]], axis=-1)
    feet_down_xz_mean = feet_down_xz.mean(axis=0)
    feet_down_y = feet_down[:, 1:2]
    feet_down_y_mean = feet_down_y.mean(axis=0)
    feet_down_y_std = feet_down_y.std(axis=0)

    """ Up Locations """

    feet_up_xz = np.concatenate([feet_up[:, 0:1], feet_up[:, 2:3]], axis=-1)
    feet_up_y = feet_up[:, 1:2]

    if len(feet_down_xz) == 0:

        """ No Contacts """

        def terr_func(Xp): return np.zeros_like(Xp)[:, :1][np.newaxis].repeat(nsamples, axis=0)

    elif type == 'flat':

        """ Flat """

        def terr_func(Xp):
            return np.zeros_like(Xp)[:, :1][np.newaxis].repeat(nsamples, axis=0) + feet_down_y_mean

    else:

        """ Terrain Heights """

        terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean)
        terr_down_y_mean = terr_down_y.mean(axis=1)
        terr_down_y_std = terr_down_y.std(axis=1)
        terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean)

        """ Fitting Error """

        terr_down_err = 0.1 * ((
            (terr_down_y - terr_down_y_mean[:, np.newaxis]) -
            (feet_down_y - feet_down_y_mean)[np.newaxis])**2)[..., 0].mean(axis=1)

        terr_up_err = (np.maximum(
            (terr_up_y - terr_down_y_mean[:, np.newaxis]) -
            (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0)**2)[..., 0].mean(axis=1)

        """ Jumping Error """

        terr_over_err = 0.0

        """ Fitting Terrain to Walking on Beam """

        terr_beam_err = 0.0

        """ Final Fitting Error """

        terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err

        """ Best Fitting Terrains """

        terr_ids = np.argsort(terr)[:nsamples]
        terr_patches = patches[terr_ids]

        def terr_basic_func(Xp): return (
            (patchfunc(terr_patches, Xp - feet_down_xz_mean) -
             terr_down_y_mean[terr_ids][:, np.newaxis]) + feet_down_y_mean)
        """ Terrain Fit Editing """

        terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
        terr_fine_func = [RBF(smooth=0.1, function='linear')
                          for _ in range(nsamples)]
        for i in range(nsamples):
            terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])

        def terr_func(Xp): return (terr_basic_func(Xp) +
                                   np.array([ff(Xp) for ff in terr_fine_func]))
    """ Get Trajectory Terrain Heights """

    root_offsets_c = global_positions[:, 0]
    root_offsets_r = (-root_rotation[:, 0] *
                      np.array([[+25, 0, 0]])) + root_offsets_c
    root_offsets_l = (-root_rotation[:, 0] *
                      np.array([[-25, 0, 0]])) + root_offsets_c

    root_heights_c = terr_func(root_offsets_c[:, np.array([0, 2])])[..., 0]
    root_heights_r = terr_func(root_offsets_r[:, np.array([0, 2])])[..., 0]
    root_heights_l = terr_func(root_offsets_l[:, np.array([0, 2])])[..., 0]

    """ Find Trajectory Heights at each Window """

    root_terrains = []
    root_averages = []
    for i in range(window, len(anim)-window, 1):
        root_terrains.append(
            np.concatenate([
                root_heights_r[:, i-window:i+window:10],
                root_heights_c[:, i-window:i+window:10],
                root_heights_l[:, i-window:i+window:10]], axis=1))
        root_averages.append(
            root_heights_c[:, i-window:i+window:10].mean(axis=1))

    root_terrains = np.swapaxes(np.array(root_terrains), 0, 1)
    root_averages = np.swapaxes(np.array(root_averages), 0, 1)

    return root_terrains, root_averages


""" Phases, Inputs, Outputs """

P, X, Y = [], [], []

for data in data_terrain:

    print('Processing Clip %s' % data)

    """ Data Types """

    if 'Flat' in data:
        type = 'flat'
    else:
        type = 'rocky'

    """ Load Data """

    anim, names, _ = BVH.load(data)
    anim.offsets *= to_meters
    anim.positions *= to_meters
    anim = anim[::2]

    """ Load Phase / Gait """

    phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
    gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]

    """ Merge Jog / Run and Crouch / Crawl """

    gait = np.concatenate([
        gait[:, 0:1],
        gait[:, 1:2],
        gait[:, 2:3] + gait[:, 3:4],
        gait[:, 4:5] + gait[:, 6:7],
        gait[:, 5:6],
        gait[:, 7:8]
    ], axis=-1)

    """ Preprocess Data """

    Pc, Xc, Yc = process_data(anim, phase, gait, type=type)

    with open(data.replace('.bvh', '_footsteps.txt'), 'r') as f:
        footsteps = f.readlines()

    """ For each Locomotion Cycle fit Terrains """

    for li in range(len(footsteps)-1):

        curr, next = footsteps[li+0].split(' '), footsteps[li+1].split(' ')

        """ Ignore Cycles marked with '*' or not in range """

        if len(curr) == 3 and curr[2].strip().endswith('*'):
            continue
        if len(next) == 3 and next[2].strip().endswith('*'):
            continue
        if len(next) < 2:
            continue
        if int(curr[0])//2-window < 0:
            continue
        if int(next[0])//2-window >= len(Xc):
            continue

        """ Fit Heightmaps """

        slc = slice(int(curr[0])//2-window, int(next[0])//2-window+1)
        H, Hmean = process_heights(anim[int(curr[0])//2-window: int(next[0])//2+window+1], type=type)

        for h, hmean in zip(H, Hmean):

            Xh, Yh = Xc[slc].copy(), Yc[slc].copy()

            """ Reduce Heights in Input/Output to Match"""

            xo_s, xo_e = ((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1
            yo_s, yo_e = 8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1
            Xh[:, xo_s:xo_e:3] -= hmean[..., np.newaxis]
            Yh[:, yo_s:yo_e:3] -= hmean[..., np.newaxis]
            Xh = np.concatenate([Xh, h - hmean[..., np.newaxis]], axis=-1)

            """ Append to Data """

            P.append(np.hstack([0.0, Pc[slc][1:-1], 1.0]).astype(np.float32))
            X.append(Xh.astype(np.float32))
            Y.append(Yh.astype(np.float32))

""" Clip Statistics """

print('Total Clips: %i' % len(X))
print('Shortest Clip: %i' % min(map(len, X)))
print('Longest Clip: %i' % max(map(len, X)))
print('Average Clip: %i' % np.mean(list(map(len, X))))

""" Merge Clips """

print('Merging Clips...')

Xun = np.concatenate(X, axis=0)
Yun = np.concatenate(Y, axis=0)
Pun = np.concatenate(P, axis=0)

print(Xun.shape, Yun.shape, Pun.shape)

print('Saving Database...')

np.savez_compressed('./data/mydatabase.npz', Xun=Xun, Yun=Yun, Pun=Pun)
