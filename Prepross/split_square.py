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
root = 'D:/MA/H_MA/'
merge_v2_root = 'D:/MA/H_MA/mergev2_h_dat/'
in_box_root = 'D:/MA/H_MA/inbox_dat/'
if not os.path.exists(in_box_root):
    os.makedirs(in_box_root)
columnindex = ['x', 'y', 'z', 'head_x', 'head_y', 'head_z']
def split_points(file):
    ## x: -250，1750，80
    ## y: -1825，-75，70
    file_name = file.split('.')[0]
    merge_v2_path = os.path.join(merge_v2_root, file)
    with open(merge_v2_path, 'rb') as f:
        points_with_head = pickle.load(f)
    points_with_head_pd = pd.DataFrame(points_with_head, columns=columnindex)
    with open(os.path.join(root, 'dict_range.dat'), 'rb') as f:
        dict_range = pickle.load(f)
    coor_range = dict_range['{}'.format(file_name)]
    idx_s = int((coor_range[0] + 250)/25)
    idx_e = int((coor_range[1] + 250)/25)
    idy_s = int((coor_range[2] + 1825) / 25)
    idy_e = int((coor_range[3] + 1825) / 25)
    for px in tqdm(range(idx_s, idx_e + 1)):
        for py in range(idy_s, idy_e + 1):
            in_box = pd.DataFrame(columns=columnindex)
            if os.path.exists(os.path.join(in_box_root, '{}_{}.dat'.format(px, py))):
                with open(os.path.join(in_box_root, '{}_{}.dat'.format(px, py)), 'rb') as f:
                    in_boxed = pickle.load(f)
                    in_box = in_box.append(in_boxed)
            box_min_x = -250 + 25 * px
            box_max_x = -250 + 25 * (px + 1)
            box_min_y = -1825 + 25 * py
            box_max_y = -1825 + 25 * (py + 1)
            in_box = in_box.append(points_with_head_pd[(points_with_head_pd['x'] >= box_min_x) &
                                                       (points_with_head_pd['x'] < box_max_x) &
                                                       (points_with_head_pd['y'] >= box_min_y) &
                                                       (points_with_head_pd['y'] < box_max_y)])
            if len(in_box) > 0:
                with open(os.path.join(in_box_root, '{}_{}.dat'.format(px, py)), "wb") as f:
                    pickle.dump(in_box, f)





files = os.listdir(merge_v2_root)
for idx, file in enumerate(files):
    #if idx > 0:
    #    break
    print("{}:  {} / {}".format(file, idx + 1, len(files)))
    split_points(file)