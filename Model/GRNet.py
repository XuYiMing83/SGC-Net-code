import torch

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling
import pytorch3d.loss
from pytorch3d.ops import sample_farthest_points, knn_points


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
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


class GRNet(torch.nn.Module):
    def __init__(self):
        super(GRNet, self).__init__()
        self.gridding = Gridding(scale=80)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 40, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(40),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(40, 80, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(80),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(80, 160, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(160),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(160, 320, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(320),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(40000, 8000),
            torch.nn.ReLU()
        )
        self.fcx1 = torch.nn.Sequential(
            torch.nn.Linear(8000, 4000),
            torch.nn.ReLU()
        )
        self.fcx2 = torch.nn.Sequential(
            torch.nn.Linear(4000, 8000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(8000, 40000),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(320, 160, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(160),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(160, 80, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(80),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(80, 40, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(40),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(40, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = GriddingReverse(scale=80)
        self.point_sampling = RandomPointSampling(n_points=3072)
        self.feature_sampling = CubicFeatureSampling()
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(2240, 2240),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(2240, 560),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(560, 140),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(140, 27)

    def forward(self, data):
        partial_cloud = data.transpose(2, 1)
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 80, 80, 80)
        # print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(pt_features_64_l)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 40000))
        # print(features.size())          # torch.Size([batch_size, 2048])
        features = self.fcx1(features)
        features = self.fcx2(features)
        pt_features_4_r = self.fc6(features).view(-1, 320, 5, 5, 5) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_r = self.dconv10(pt_features_32_r) + pt_features_64_l
        # print(pt_features_64_r.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev(pt_features_64_r.squeeze(dim=1))
        # print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, 3072, 320)
        # print(point_features_32.size()) # torch.Size([batch_size, 2048, 256])
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, 3072, 640)
        # print(point_features_16.size()) # torch.Size([batch_size, 2048, 512])
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, 3072, 1280)
        # print(point_features_8.size())  # torch.Size([batch_size, 2048, 1024])
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 448])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 112])
        point_offset = self.fc14(point_features).view(-1, 27648, 3)
        # print(point_features.size())    # torch.Size([batch_size, 16384, 3])
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 9, 1).view(-1, 27648, 3) + point_offset
        # print(dense_cloud.size())       # torch.Size([batch_size, 16384, 3])

        return sparse_cloud, dense_cloud

from emdloss import emd_module as emdm
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.emd = emdm.emdModule()

    def forward(self, coarse, fine, gt, alpha, eps=0.005, iters=50):  # 0.005, 50 for training  0.002, 10000 is testing
        gt = gt.transpose(2, 1)

        # gt_ds = sample_farthest_points(points=gt, K=coarse.shape[1])[0]
        # dist, _ = self.emd(coarse, gt_ds, eps, iters)
        # loss_coarse = torch.sqrt(dist).mean()

        loss_fine, _ = pytorch3d.loss.chamfer_distance(fine, gt)
        loss_coarse, _ = pytorch3d.loss.chamfer_distance(coarse, gt)

        loss = loss_coarse + alpha * loss_fine
        return loss, loss_coarse, loss_fine