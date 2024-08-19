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
point_filtered_root = 'D:/MA/H_MA/dumpfiles_2m_H/'
point_root = 'E:/Studienarbeit/ikg_MA-main/xym/ikg/1_parse/parse_dumpfiles_all'
save_path = 'D:/MA/H_MA/mergev2_h/'
save_path_dat = 'D:/MA/H_MA/mergev2_h_dat/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path_dat):
    os.makedirs(save_path_dat)

hildesheim_x_offset = 564546
hildesheim_y_offset = 5778458

files = os.listdir(point_filtered_root)
for idx, file in enumerate(tqdm(files)):

    path_parse_dumpfile = os.path.join(point_filtered_root, file)
    if int(file.split('_')[3]) == 1:
        with open(os.path.join(point_root, file, "head_info.dat"), 'rb') as f:
            header = pickle.load(f)
        base_x, base_y, base_z = header["original_x"], header["original_y"], header["original_z"]
        with open(os.path.join(point_filtered_root, file, "points.dat"), 'rb') as f:
            point_filtered_1 = pickle.load(f)
        point_filtered_1[:, 0] += base_x
        point_filtered_1[:, 1] += base_y
        point_filtered_1[:, 2] += base_z
        point_filtered_1[:, 3] += base_x
        point_filtered_1[:, 4] += base_y
        point_filtered_1[:, 5] += base_z
    if int(file.split('_')[3]) == 2:
        with open(os.path.join(point_root, file, "head_info.dat"), 'rb') as f:
            header = pickle.load(f)
        base_x, base_y, base_z = header["original_x"], header["original_y"], header["original_z"]
        with open(os.path.join(point_filtered_root, file, "points.dat"), 'rb') as f:
            point_filtered_2 = pickle.load(f)
        point_filtered_2[:, 0] += base_x
        point_filtered_2[:, 1] += base_y
        point_filtered_2[:, 2] += base_z
        point_filtered_2[:, 3] += base_x
        point_filtered_2[:, 4] += base_y
        point_filtered_2[:, 5] += base_z
        point_filtered_merge = np.concatenate([point_filtered_1, point_filtered_2])
        point_filtered_merge[:, 0] -= hildesheim_x_offset
        point_filtered_merge[:, 1] -= hildesheim_y_offset
        point_filtered_merge[:, 3] -= hildesheim_x_offset
        point_filtered_merge[:, 4] -= hildesheim_y_offset
        #write_points_dddx2(point_filtered_merge, os.path.join(save_path, '{}.ply'.format(file.split('_')[0]+'_'+file.split('_')[1])))
        with open(os.path.join(save_path_dat, '{}.dat'.format(file.split('_')[0]+'_'+file.split('_')[1])), "wb") as f:
            pickle.dump(point_filtered_merge, f)
