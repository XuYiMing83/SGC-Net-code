import numpy as np
import struct
import pickle
from collections import Counter
from skimage import io
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import time
import math
from PIL import Image, ImageDraw, ImageFont
import os
import csv
import open3d as o3d
import os
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
import ransac
from ply import write_points_dddx2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points

SUB_root = 'D:/MA/D_MA/gt'
train_root = 'D:/MA/D_MA/train'
gt_root = 'D:/MA/D_MA/gt_norm'
training_root = 'D:/MA/D_MA/train_norm'
num_gt = 20480
num_training = 14000
z_scalar = 3
if not os.path.exists(gt_root):
    os.makedirs(gt_root)
if not os.path.exists(training_root):
    os.makedirs(training_root)
def load_plydata(address):
    #plydata = PlyData.read(address)
    _data = PlyData.read(address).elements[0].data
    plane_tango = None
    if len(_data) == 0:
        return None
    if len(_data[1]) == 6:
        plane_tango = np.stack((_data['x'], _data['y'], _data['z'],
                                _data['scalar_head_x'],
                                _data['scalar_head_y'],
                                _data['scalar_head_z'],
                                ),axis=1)
    return plane_tango

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]
a=2.5
files = os.listdir(SUB_root)
for idx, file in enumerate(tqdm(files)):
    #print("{}:  {} / {}".format(file, idx + 1, len(files)))
    car_x = float(file.split('_')[0])
    car_y = float(file.split('_')[1].split('.')[0] + '.' + file.split('_')[1].split('.')[1])

    # gt
    plys = load_plydata(os.path.join(SUB_root, file))[:, :3]
    if abs(max(plys[:, 2]) - min(plys[:, 2])) > 2.666:
        print(min(plys[:, 2]))
        idxs_z = np.where(plys[:, 2] - min(plys[:, 2]) < 2.666)[0]
        plys = plys[idxs_z]
    mean_z0 = round(plys.mean(0)[2], 6)
    plys[:, 0] = (plys[:, 0] - car_x) / 4
    plys[:, 1] = (plys[:, 1] - car_y) / 4
    plys[:, 2] = a * (plys[:, 2] - mean_z0) / 4

    idxs_z_p = np.where(plys[:, 2] < 1)[0]
    idxs_z_m = np.where(plys[:, 2] > -1)[0]
    idxs_z = idxs_z_p[np.in1d(idxs_z_p, idxs_z_m)]
    plys = plys[idxs_z]
    plys[:, 2] = (plys[:, 2]) * 4 / a + mean_z0
    mean_z = round(abs(max(plys[:, 2]) + min(plys[:, 2])) / 2, 6)
    plys[:, 2] = z_scalar * (plys[:, 2] - mean_z) / 4
    # plys = resample_pcd(plys, 46080)

    name = '{}_{}_{}.ply'.format(car_x, car_y, mean_z)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plys)
    o3d.io.write_point_cloud(os.path.join(gt_root,name), pcd)

    # training
    plyt = load_plydata(os.path.join(train_root, file))[:, :3]
    plyt[:, 0] = (plyt[:, 0] - car_x) / 4
    plyt[:, 1] = (plyt[:, 1] - car_y) / 4
    plyt[:, 2] = a * (plyt[:, 2] - mean_z0) / 4

    idxt_z_p = np.where(plyt[:, 2] < 1)[0]
    idxt_z_m = np.where(plyt[:, 2] > -1)[0]
    idxt_z = idxt_z_p[np.in1d(idxt_z_p, idxt_z_m)]
    plyt = plyt[idxt_z]
    plyt[:, 2] = (plyt[:, 2]) * 4 / a + mean_z0
    plyt[:, 2] = z_scalar * (plyt[:, 2] - mean_z) / 4
    idxt_z = np.where(plyt[:, 2] < 1)[0]
    plyt = plyt[idxt_z]
    # plyt = resample_pcd(plyt, 34000)


    pcdt = o3d.geometry.PointCloud()
    pcdt.points = o3d.utility.Vector3dVector(plyt)
    o3d.io.write_point_cloud(os.path.join(training_root, name), pcdt)


