import open3d
import torch
import numpy as np
from plyfile import PlyData, PlyElement

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

def _tensor_get_open3d_ptcloud(tensor):
    tensor = tensor.squeeze().cpu().numpy()
    ptcloud = open3d.geometry.PointCloud()
    ptcloud.points = open3d.utility.Vector3dVector(tensor)

    return ptcloud

def _np_get_open3d_ptcloud(array):
    ptcloud = open3d.geometry.PointCloud()
    ptcloud.points = open3d.utility.Vector3dVector(array)

    return ptcloud

def calculate_fscore(pr, gt, th):
    """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    if type(gt) == open3d.geometry.PointCloud and type(pr) == open3d.geometry.PointCloud:
        pass
    elif type(gt) == np.ndarray and type(pr) == np.ndarray:
        gt = _np_get_open3d_ptcloud(gt)
        pr = _np_get_open3d_ptcloud(pr)
    elif type(gt) == torch.Tensor and type(pr) == torch.Tensor:
        gt = _tensor_get_open3d_ptcloud(gt)
        pr = _tensor_get_open3d_ptcloud(pr)
    else:
        raise ValueError('The input type must be open3d.geometry.PointCloud, np.ndarray or torch.Tensor')


    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall

from emdloss import emd_module as emd
def emd_error(pr, gt):
    emd_error = emd.emdModule()
    gt = torch.tensor(gt).cuda().float()
    pr = torch.tensor(pr).cuda().float()
    dist, _ = emd_error(pr, gt, eps=0.002, iters=500)
    loss = torch.sqrt(dist).mean()
    return loss

import pytorch3d.loss
def cd_error(pr, gt):
    gt = torch.tensor(gt).cuda().float()
    pr = torch.tensor(pr).cuda().float()
    loss, _ = pytorch3d.loss.chamfer_distance(pr, gt)
    return loss

def back_z_coor(pr, gt):
    gt[:, :, 2] = gt[:, :, 2] / 3
    pr[:, :, 2] = pr[:, :, 2] / 3
    return pr, gt

# gt = load_plydata(r'D:\MA\D_MA\20480\stest_resultSGC01re\81_gt.ply')
# gt[:, 2] = gt[:, 2] / 3
# # pr = load_plydata(r'D:\MA\D_MA\20480\stest_resultSGC01re\81_training.ply')
# pr = load_plydata(r'D:\MA\D_MA\20480\stest_resultSGC01re\81_fine_new.ply')
# # pr = load_plydata('D:/MA/D_MA/20480/data_all/fine_sub_test_training01/418.075226_-350.308858_126.175243.ply')
# pr[:, 2] = pr[:, 2] / 3
# fscore, precision, recall = calculate_fscore(gt, pr, 0.01)
# print(fscore, precision, recall)