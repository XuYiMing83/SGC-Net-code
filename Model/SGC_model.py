import torch.nn as nn
import torch
import torch.nn.functional as F
# import numpy as np
from torch.autograd import Variable
import pytorch3d.loss
from pytorch3d.ops import sample_farthest_points, knn_points
import itertools

class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        #print(pred_cloud.size(), partial_cloud.size())
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling
class SGC(torch.nn.Module):
    def __init__(self, convlist, test=False):
        super(SGC, self).__init__()
        self.fc_p = convlist[4]
        self.cube0 = convlist[0]
        self.cube1 = convlist[1]
        self.cube2 = convlist[2]
        self.cube3 = convlist[3]
        self.test = test

        a = torch.linspace(-0.1, 0.1, steps=3, dtype=torch.float).view(1, 3).expand(3, 3).reshape(1, -1)
        b = torch.linspace(-0.1, 0.1, steps=3, dtype=torch.float).view(3, 1).expand(3, 3).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.gridding = Gridding(scale=80)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, convlist[0], kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(convlist[0]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(convlist[0], convlist[1], kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(convlist[1]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(convlist[1], convlist[2], kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(convlist[2]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(convlist[2], convlist[3], kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(convlist[3]),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(convlist[4], convlist[4]//5),
            torch.nn.ReLU()
        )
        self.fcx1 = torch.nn.Sequential(
            torch.nn.Linear(convlist[4]//5, convlist[4]//10),
            torch.nn.ReLU()
        )
        self.fcx2 = torch.nn.Sequential(
            torch.nn.Linear(convlist[4]//10, convlist[4]//5),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(convlist[4]//5, convlist[4]),
            torch.nn.ReLU(),
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(convlist[3], convlist[2], kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(convlist[2]),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(convlist[2], convlist[1], kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(convlist[1]),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(convlist[1], convlist[0], kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(convlist[0]),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(convlist[0], 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=80)

        self.point_sampling = RandomPointSampling(n_points=3072)
        self.feature_sampling = CubicFeatureSampling()
        self.mlps = torch.nn.Sequential(
            torch.nn.Linear(8 * (self.cube0 + self.cube1 + self.cube2), 560),
            torch.nn.GELU(),
            torch.nn.Linear(560, 560),
            torch.nn.GELU(),
            torch.nn.Linear(560, 280)
        )

        self.folding1 = nn.Sequential(
            nn.Conv1d(280 + 3 + 2, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(280 + 3 + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
        )

    def forward(self, data):
        B = data.size()[0]
        if data.size()[1] != 18500 and self.test is True:
            data = sample_farthest_points(points=data.transpose(2, 1), K=18500)[0].transpose(2, 1)
        partial_cloud = data.transpose(2, 1)
        # coarse
        pt_features_80_l = self.gridding(partial_cloud).view(-1, 1, 80, 80, 80)
        pt_features_40_l = self.conv1(pt_features_80_l)
        pt_features_20_l = self.conv2(pt_features_40_l)
        pt_features_10_l = self.conv3(pt_features_20_l)
        pt_features_5_l = self.conv4(pt_features_10_l)
        features = self.fc5(pt_features_5_l.view(-1, self.fc_p))
        features = self.fcx1(features)
        features = self.fcx2(features)
        pt_features_5_r = self.fc6(features).view(-1, self.cube3, 5, 5, 5) + pt_features_5_l
        pt_features_10_r = self.dconv7(pt_features_5_r) + pt_features_10_l
        pt_features_20_r = self.dconv8(pt_features_10_r) + pt_features_20_l
        pt_features_40_r = self.dconv9(pt_features_20_r) + pt_features_40_l
        pt_features_80_r = self.dconv10(pt_features_40_r) + pt_features_80_l
        sparse_cloud_r = self.gridding_rev(pt_features_80_r.squeeze(dim=1))
        sparse_cloud = self.point_sampling(sparse_cloud_r)

        #fine
        point_features_40 = self.feature_sampling(sparse_cloud, pt_features_40_r).view(-1, 3072, 8 * self.cube0)
        point_features_20 = self.feature_sampling(sparse_cloud, pt_features_20_r).view(-1, 3072, 8 * self.cube1)
        point_features_10 = self.feature_sampling(sparse_cloud, pt_features_10_r).view(-1, 3072, 8 * self.cube2)
        point_features_cube = torch.cat([point_features_40, point_features_20, point_features_10], dim=2)
        point_features = self.mlps(point_features_cube)  # B, 3072, 280

        grid = self.folding_seed.view(1, 2, 9).expand(B, 2, 9).cuda().repeat(1, 1, 3072)
        point_feature_feat = point_features.unsqueeze(dim=2).repeat(1, 1, 9, 1).view(-1, 27648, 280).transpose(2, 1)
        point_feat = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 9, 1).view(-1, 27648, 3)
        # cube_feat = point_features_cube.unsqueeze(dim=2).repeat(1, 1, 9, 1).view(-1, 27648, 2240).transpose(2, 1)
        feat = torch.cat((point_feature_feat, point_feat.transpose(2, 1)), 1)  # B, 283, 27648
        point_offset = self.folding1(torch.cat([grid, feat], dim=1))
        point_offset = self.folding2(torch.cat([point_offset, feat], dim=1)).transpose(2, 1)
        dense_cloud = point_feat + point_offset

        return sparse_cloud, dense_cloud



from emdloss import emd_module as emdm
from extensions.gridding_loss import GriddingLoss
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.emd = emdm.emdModule()
        self.gridding_loss = GriddingLoss(scales=[80], alphas=[1])

    def forward(self, coarse, fine, gt, alpha):  # 0.005, 50 for training  0.002, 10000 is testing
        gt = gt.transpose(2, 1)
        loss_fine, _ = pytorch3d.loss.chamfer_distance(fine, gt)
        loss_corse, _ = pytorch3d.loss.chamfer_distance(coarse, gt)
        loss = loss_corse + alpha * loss_fine
        return loss, loss_corse, loss_fine
