import numpy as np
import pickle
from tqdm import tqdm
import ransac
from sklearn import linear_model
import random
import os
from plyfile import PlyData
from sklearn.neighbors import KDTree
import pandas as pd
from ply import write_points_dddx2
import open3d as o3d
CCPath = "E:/cloudcompare/CloudCompare"
car_mesh_root = 'D:/MA/car_dnorm/'
in_box_root = 'D:/MA/D_MA/sub_inbox_ply/'
edge_path = 'D:/MA/D_MA/all_point.dat'
hildesheim_x_offset = 564546
hildesheim_y_offset = 5778458

train_root = 'D:/MA/H_MA/data/train/'
car_root = 'D:/MA/H_MA/data/cars/'
original_root = 'D:/MA/H_MA/data/original/'
sub_root = 'D:/MA/H_MA/data/sub/'
if not os.path.exists(train_root):
    os.makedirs(train_root)
if not os.path.exists(car_root):
    os.makedirs(car_root)
if not os.path.exists(original_root):
    os.makedirs(original_root)
if not os.path.exists(sub_root):
    os.makedirs(sub_root)

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

def wide_randm(w):
    a = random.uniform(1.65, 2)
    return a / w

def get_edge_points(path):
    with open(path, "rb") as f:
        edge_points = pickle.load(f)
    return edge_points

def dir_path_mesh():
    files_mesh = os.listdir(car_mesh_root)
    num = len(files_mesh)
    idx = random.randint(0, num - 1)
    return os.path.join(car_mesh_root, files_mesh[idx])

def read_rotation_car_mesh(car_mesh_path):
    mesh = o3d.io.read_triangle_mesh(car_mesh_path)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh = mesh.translate(-1 * np.min(mesh.vertices, axis=0))
    car_wide = np.max(mesh.vertices, axis=0)[0]
    scale_wide = wide_randm(car_wide)
    mesh.scale(scale_wide, center=(0, 0, 0))
    mesh = mesh.translate([-np.max(mesh.vertices, axis=0)[0] / 2, -np.max(mesh.vertices, axis=0)[1] / 2, 0])
    return mesh

def gt_gen(car_x_h, car_y_h):
    neighbor_8 = ([-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1])
    columnindex = ['x', 'y', 'z', 'head_x', 'head_y', 'head_z']
    box_8 = pd.DataFrame(columns=columnindex)
    x_position = int((car_x_h + 250) / 25)
    y_position = int((car_y_h + 1825) / 25)
    for neighbor in neighbor_8:
        box_idx = neighbor + np.array([x_position, y_position])
        if os.path.exists(os.path.join(in_box_root, '{}_{}.ply'.format(int(box_idx[0]), int(box_idx[1])))):
            ply_box = load_plydata(os.path.join(in_box_root, '{}_{}.ply'.format(int(box_idx[0]), int(box_idx[1]))))
            ply_box = pd.DataFrame(ply_box, columns=columnindex)
            box_8 = box_8.append(ply_box)
    points = box_8.values[:, 0:3]
    c1 = np.where(points[:, 0] < car_x_h + 4)[0]
    c2 = np.where(points[:, 0] > car_x_h - 4)[0]
    cx = c1[np.in1d(c1, c2)]
    c3 = np.where(points[:, 1] < car_y_h + 4)[0]
    c4 = np.where(points[:, 1] > car_y_h - 4)[0]
    cy = c3[np.in1d(c3, c4)]
    c = cx[np.in1d(cx, cy)]
    square_point = box_8.values[c]
    if square_point.shape[0] < 15000:
        return False
    write_points_dddx2(square_point,
                     'D:/MA/D_MA/w/gt/{}_{}.ply'.format(round(car_x_h, 6), round(car_y_h, 6)))
    return True

def head_angle(car_x_h, car_y_h):
    gt_name = 'D:/MA/D_MA/w/gt/{}_{}.ply'.format(round(car_x_h, 6), round(car_y_h, 6))
    ply = load_plydata(gt_name)
    X = ply[:, 3][:, np.newaxis]
    y = ply[:, 4]
    rs = linear_model.RANSACRegressor()
    rs.fit(X, y)
    k = rs.estimator_.coef_
    return np.arctan(k)[0]

def car_drive_and_stop(car_x_h, car_y_h, mesh):
    gt_name = 'D:/MA/D_MA/w/gt/{}_{}.ply'.format(round(car_x_h, 6), round(car_y_h, 6))
    ply = load_plydata(gt_name)
    model = ransac.RANSAC()
    m, inlier_mask, _ = model.run(ply[:, :3], inlier_thres=0.6, max_iterations=20)
    a, b, c, d = m
    car_z_h = -(a * car_x_h + b * car_y_h + d) / c
    rot_h = head_angle(car_x_h, car_y_h)
    R = mesh.get_rotation_matrix_from_xyz((0, 0, rot_h + np.pi/2))
    mesh.rotate(R, center=(0, 0, 0))
    if c >= 0:
        cross = np.cross(np.array([0, 0, 1]), np.array([a, b, c]) / ((a ** 2 + b ** 2 + c ** 2) ** 0.5))
    else:
        cross = np.cross(np.array([0, 0, 1]), np.array([-a, -b, -c]) / ((a ** 2 + b ** 2 + c ** 2) ** 0.5))
    c_s = np.arcsin(np.linalg.norm(cross)) / np.linalg.norm(cross)
    cross_ = cross * c_s
    R = mesh.get_rotation_matrix_from_axis_angle(cross_.T)
    mesh.rotate(R, center=(0, 0, 0))
    mesh = mesh.translate((car_x_h, car_y_h, car_z_h))
    o3d.io.write_triangle_mesh('D:/MA/D_MA/w/cars/{}_{}.ply'.format(round(car_x_h, 6), round(car_y_h, 6)), mesh)

    light_d = ply[:, 3:6] - ply[:, 0:3]
    light = np.concatenate([ply[:, 0:3], light_d], axis=1)
    rays = o3d.core.Tensor(light, dtype=o3d.core.Dtype.Float32)
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    ans = scene.cast_rays(rays)
    ansid = ans['t_hit'].numpy()

    train_circle = ply[np.where([ansid == np.inf])[1]]
    write_points_dddx2(train_circle, 'D:/MA/D_MA/w/train/{}_{}.ply'.format(round(car_x_h, 6), round(car_y_h, 6)))

def dist_randm():
    a = random.uniform(0.2, 0.4)
    return a

def car_xy_h_finder(px, py, mesh):
    px_h = px - hildesheim_x_offset
    py_h = py - hildesheim_y_offset
    px_position = int((px_h + 250) / 25)
    py_position = int((py_h + 1825) / 25)
    if os.path.exists(os.path.join(in_box_root, '{}_{}.ply'.format(int(px_position), int(py_position)))) is False:
        return None, None
    box = load_plydata(os.path.join(in_box_root, '{}_{}.ply'.format(int(px_position), int(py_position))))
    tree = KDTree(box[:, :2], leaf_size=30)
    dists, ind = tree.query([[px_h, py_h]], k=30)
    update_values = box[ind[0]].mean(axis=0)
    x_up_head = update_values[3]
    y_up_head = update_values[4]
    direction = np.array([x_up_head - px_h, y_up_head - py_h])
    direction = direction / np.linalg.norm(direction)
    # move_dist = 0.12 + np.max(mesh.vertices, axis=0)[0]
    move_dist = dist_randm()
    car_h = np.array([px_h, py_h]) + direction * move_dist
    car_x_h = car_h[0]
    car_y_h = car_h[1]
    return car_x_h, car_y_h

i = 0
edge_points = get_edge_points(edge_path)
for points in tqdm(edge_points):
    i += 1
    # if i < 2482:
    #     continue
    # px, py = points
    px, py = edge_points[234]
    # mesh_path = dir_path_mesh()
    mesh_path = 'D:/MA/car_dnorm/1c1bd2dcbb13aa5a6b652ed61c4ad126.ply'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    car_x_h, car_y_h = car_xy_h_finder(px, py, mesh)
    if car_y_h is None:
        continue
    gt = gt_gen(car_x_h, car_y_h)
    if gt is False:
        continue
    car_drive_and_stop(car_x_h, car_y_h, mesh)



