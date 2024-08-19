import os
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import random
warnings.filterwarnings('ignore')
scalar = 0.95

def load_plydata(address):
    #plydata = PlyData.read(address)
    _data = PlyData.read(address).elements[0].data
    plane_tango = None
    if len(_data) == 0:
        return None
    if len(_data[1]) == 3:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z']), axis=1)
    if len(_data[1]) == 6:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z'], _data['scalar_head_x'],
                                _data['scalar_head_y'], _data['scalar_head_z']), axis=1)
        plane_tango = plane_tango[:, :3]
    return plane_tango

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def norm_square(points, file, debug=False):
    car_x = float(file.split('_')[0])
    car_y = float(file.split('_')[1].split('.')[0] + '.' + file.split('_')[1].split('.')[1])
    if debug is True:
        if abs(max(points[:, 2]) - min(points[:, 2])) > 2.666:
            idxs_z = np.where(points[:, 2] - min(points[:, 2]) < 2.666)[0]
            points = points[idxs_z]

        points[:, 0] = (points[:, 0] - car_x) / 4
        idxs_x_p = np.where(points[:, 0] < 1)[0]
        idxs_x_m = np.where(points[:, 0] > -1)[0]
        idxs_x = idxs_x_p[np.in1d(idxs_x_p, idxs_x_m)]
        points = points[idxs_x]
        points[:, 0] = points[:, 0] * 4 + car_x
        points[:, 1] = (points[:, 1] - car_y) / 4
        idxs_y_p = np.where(points[:, 1] < 1)[0]
        idxs_y_m = np.where(points[:, 1] > -1)[0]
        idxs_y = idxs_y_p[np.in1d(idxs_y_p, idxs_y_m)]
        points = points[idxs_y]
        points[:, 1] = points[:, 1] * 4 + car_y
    points[:, 0] = (points[:, 0] - car_x) / 4
    points[:, 1] = (points[:, 1] - car_y) / 4
    mean_z = round(abs(max(points[:, 2]) + min(points[:, 2])) / 2, 6)
    points[:, 2] = 3 * (points[:, 2] - mean_z) / 4
    if min(points[:, 0]) < -1 or max(points[:, 0]) > 1 or min(points[:, 1]) < -1 \
            or max(points[:, 1]) > 1 or min(points[:, 2]) < -1 or max(points[:, 2]) > 1:
        print('x:', min(points[:, 0]), 'to', max(points[:, 0]))
        print('y:', min(points[:, 1]), 'to', max(points[:, 1]))
        print('z:', min(points[:, 2]), 'to', max(points[:, 2]))
        raise ValueError('The x * y size of the input point cloud cannot exceed 8m * by 8m, '
                         'and the height difference cannot exceed 2.6m.')
    if points.shape[0] > 200000:
        points = resample_pcd(points, 200000)
    return points, mean_z

def random_rot_matrx(random_rotation_mirror=True):
    rot_matrx = np.array([[[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          [[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]],
                          [[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]],
                          [[0, 1, 0],
                           [-1, 0, 0],
                           [0, 0, 1]]])
    if random_rotation_mirror:
        idx = random.randint(0, 3)
    else:
        idx = 0
    return rot_matrx[idx]

def random_mirror_matrx(random_rotation_mirror=True):
    rot_matrx = np.array([[[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          [[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]],
                          [[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]]])
    if random_rotation_mirror:
        idx = random.randint(0, 2)
    else:
        idx = 0
    return rot_matrx[idx]

class PointsDataSet(Dataset):
    def __init__(self, root, training, gt, random_rotation_mirror=True, test_debug=False):
        self.pcd_file = [file for file in os.listdir(os.path.join(root, training))]
        self.root = root
        self.training = training
        self.debug = test_debug
        if gt is not None:
            self.gt = gt
        else:
            self.gt = training
        self.point_dict = dict()
        self.random_rotation_mirror = random_rotation_mirror
        for file in tqdm(self.pcd_file):
            training_path = os.path.join(self.root, self.training, file)
            gt_path = os.path.join(self.root, self.gt, file)
            training_ = load_plydata(training_path)
            gt_ = load_plydata(gt_path)
            file_ = file
            if self.gt == self.training:
                training_, mean_z = norm_square(training_, file, self.debug)
                gt_ = training_
                file_ = [file, mean_z]
            training_ = training_ * scalar
            gt_ = gt_ * scalar
            self.point_dict['{}'.format(file)] = [training_, gt_, file_]


    def __getitem__(self, index):
        filename = self.pcd_file[index]
        rot_matrx = random_rot_matrx(self.random_rotation_mirror)
        mirror_matrx = random_mirror_matrx(self.random_rotation_mirror)
        training, gt, file = self.point_dict['{}'.format(filename)]
        training = torch.from_numpy((mirror_matrx @ (rot_matrx @ training.T)).T).transpose(1, 0)
        gt = torch.from_numpy((mirror_matrx @ (rot_matrx @ gt.T)).T).transpose(1, 0)


        return training, gt, file

    def __len__(self):
        return len(self.pcd_file)