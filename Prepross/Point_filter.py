# -*- coding: utf-8 -*-
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
import os
from plyfile import PlyData, PlyElement

import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
from ply import write_points_dddx2

import warnings
warnings.filterwarnings('ignore')

point_root = 'E:/Studienarbeit/ikg_MA-main/xym/ikg/1_parse/parse_dumpfiles_all'
root = 'D:/MA/H_MA/'
save_root = root + 'dumpfiles_2m_h/'

def load_file(name, path):
    with open(os.path.join(path, "{}.dat".format(name)), 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def save_file(data, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def generate_depth_image(path):
    coor = load_file('coordinate', path)
    head = load_file('head', path)

    tmp = coor - head
    tmp = tmp**2
    tmp = tmp.sum(axis=2)
    dis = np.sqrt(tmp)
    return dis

def run(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    coor = load_file("coordinate", path)
    head = load_file('head', path)
    coor_filtered = np.zeros(coor.shape)
    rows, columns, C = coor.shape
    depth_image = generate_depth_image(path)
    ground = np.zeros((rows, columns, 3))
    distance = np.expand_dims(depth_image, 2).repeat(3, axis=2)
    coor = np.where(distance < 15, coor, ground)
    for i in tqdm(range(columns)):
        ref_z = np.mean(coor[rows // 2 - 300:rows // 2 + 300, i, 2])
        ref_z_ = np.ones((rows, 3)) * ref_z
        coor_ = np.expand_dims(coor[:, i, 2], 1).repeat(3, axis=1)
        coor_filtered[:, i, :] = np.where(coor_ > ref_z_ + 1.85, 0, coor[:, i, :])
        coor_filtered[:, i, :] = np.where(coor_ < ref_z_ - 0.35, 0, coor_filtered[:, i, :])
    # norm = load_file('normal', path)
    # norm = np.where(coor_filtered != 0, norm, 0)
    head = np.where(coor_filtered != 0, head, 0)
    point_with_head = np.concatenate([coor_filtered, head], axis=2)
    point_filtered = point_with_head.reshape((-1, 6))[np.nonzero(np.sum(point_with_head.reshape((-1, 6)), axis=1))]

    write_points_dddx2(point_filtered, os.path.join(save_path, 'points.ply'))

    # io.imsave(os.path.join(save_path, 'norm_filtered.png'), norm)
    # with open(os.path.join(save_path, 'points.dat'), "wb") as f:
        # pickle.dump(point_filtered, f)
    # with open(os.path.join(save_path, 'points_dump_with_head.dat'), "wb") as f:
    #     pickle.dump(point_with_head, f)






files = os.listdir(point_root)
for idx, file in enumerate(files):
    print("{}:  {} / {}".format(file, idx + 1, len(files)))
    if idx > 0:
        break
    path_parse_dumpfile = os.path.join(point_root, file)
    save_path = os.path.join(save_root, file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    run(path_parse_dumpfile, save_path)