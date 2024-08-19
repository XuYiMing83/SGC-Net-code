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
in_box_ply_root = 'D:/MA/H_MA/inbox_ply/'
if not os.path.exists(in_box_ply_root):
    os.makedirs(in_box_ply_root)
columnindex = ['x', 'y', 'z', 'head_x', 'head_y', 'head_z']

def split_points_ply(file):
    file_name = file.split('.')[0]
    with open(os.path.join(in_box_root, file), 'rb') as f:
        in_box = pickle.load(f)
    if in_box.values.shape[0] <= 300:
        return None
    write_points_dddx2(in_box.values, os.path.join(in_box_ply_root, '{}.ply'.format(file_name)))


files = os.listdir(in_box_root)
for idx, file in enumerate(tqdm(files)):
    #if idx < 700:
     #   continue
    # print("{}:  {} / {}".format(file, idx + 1, len(files)))
    split_points_ply(file)