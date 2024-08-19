import os
import random
import numpy as np

import geopandas
import pandas
from tqdm import tqdm
import json
import shapely
import pickle
from shapely import geometry
from shapely.geometry import Point, Polygon, shape, MultiPoint, LineString, mapping
random.seed(255)
training_data_num = 6300
all_point = None
edge_file = 'D:/MA/c/curbs/20190919_hildesheim_adjustment_curbs.json'
edge = geopandas.read_file(edge_file)
LSL = edge.geometry.length
indx = np.where(LSL.values >= 8.5)[0]
long_edge = edge.geometry[indx].reset_index(drop=True)

temp = (long_edge.length.values.sum()-long_edge.shape[0] * 10) / training_data_num
for i in range(long_edge.shape[0]):
    line = long_edge[i]
    start_dis = 4.1
    end_dis = line.length - 4.1
    for ran in range(int((line.length - 10)/temp + 1)):
        a = random.uniform(start_dis, end_dis)
        inter_p = np.array(line.interpolate(a))[np.newaxis, :]
        if all_point is None:
            all_point = inter_p
        else:
            all_point = np.concatenate([all_point, inter_p], axis=0)
with open('D:/MA/D_MA/all_point.dat', "wb") as f:
    pickle.dump(all_point, f)
print(all_point.shape)