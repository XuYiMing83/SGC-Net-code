import os
from Model.dataset import PointsDataSet
from Model.SGC_model import SGC
from tqdm import tqdm
import open3d as o3d
from Prepross.ply import write_points_ddd, write_points_dddf
from pytorch3d.ops import sample_farthest_points, knn_points, ball_query
from evaluation import *
all_root = ''
model_path = ''
device = 'cuda'

print('loading data ...')
scalar = 0.95  # don't change it
test_dataset = PointsDataSet(root=all_root, gt='test_gt', training='test_training', random_rotation_mirror=False)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)
model_loader = torch.load(model_path, map_location='cpu')

CNN3D_dflt = 40
convn_list = list([CNN3D_dflt, CNN3D_dflt * 2, CNN3D_dflt * 4, CNN3D_dflt * 8, CNN3D_dflt * 8 * 5 * 5 * 5])
# Model = GRNet()
# Model = PCN()
Model = SGC(convn_list)
# Model.load_state_dict(model_loader)
Model.load_state_dict(model_loader['model_state_dict'])
Model = Model.to(device)
Model.eval()
cd_ = 0
fscore_ = 0
count = 0
precision_ = 0
recall_ = 0

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

with torch.no_grad():
    for batch_id, (training, gt, file_name) in tqdm(enumerate(test_dl), total=len(test_dl), smoothing=0.9):

        training = training.to(device).float()
        coarse, fine = Model(training)

        post_ = None
        for (id_p, points) in enumerate(fine):
            points = points.to(device).float().unsqueeze(dim=0)
            training_p = training[id_p].unsqueeze(dim=0).transpose(2, 1)
            # points[:, :, 2] = points[:, :, 2] / 3
            # training_p[:, :, 2] = training_p[:, :, 2] / 3
            # dists, idx, nn = knn_points(training_p, points, K=2)
            dists, idx, nn = ball_query(training_p, points, radius=0.02)
            idx = set(idx.view(1, -1, 1).squeeze().cpu().detach().numpy())
            points_idx = list(np.arange(0, 20480, 1))
            build_points_idx = [i for i in points_idx if i not in idx]
            points_in = points.cpu().detach().numpy()[0][build_points_idx, :]
            training_p = training_p.cpu().detach().numpy()[0]
            post = np.concatenate((training_p, points_in), axis=0)
            post = resample_pcd(post, 27648)
            post = post[np.newaxis, :]
            # post[:, :, 2] = post[:, :, 2] * 3
            if post_ is None:
                post_ = post
            else:
                post_ = np.concatenate((post_, post), axis=0)

        # a = fine.cpu().detach().numpy()[0]/scalar
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(a)
        # o3d.io.write_point_cloud(os.path.join('E:/model', '{}_fine.ply'.format(batch_id)), pcd)
        # a = np.array(gt.transpose(2, 1).cpu())[0]/scalar
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(a)
        # o3d.io.write_point_cloud(os.path.join('E:/model', '{}_gt.ply'.format(batch_id)), pcd)

        # generated_scene = fine.cpu().detach().numpy() / scalar
        generated_scene = post_ / scalar
        gt = gt.transpose(2, 1).numpy() / scalar
        generated_scene, gt = back_z_coor(generated_scene, gt)

        cd_scene = cd_error(generated_scene, gt)
        cd_scene = float(cd_scene.cpu().detach())


        B = generated_scene.shape[0]
        fscore = 0
        precision = 0
        recall = 0
        for (idx, points) in enumerate(generated_scene):
            fscore0, precision0, recall0 = calculate_fscore(generated_scene[idx], gt[idx], 0.01)
            fscore += fscore0
            precision += precision0
            recall += recall0

        fscore = fscore / B
        precision = precision / B
        recall = recall / B
        # print(cd_scene, fscore, precision, recall)
        cd_ += cd_scene
        fscore_ += fscore
        precision_ += precision
        recall_ += recall
        count += 1
loss_cd_test_average = cd_ / count
loss_fscore_test_average = fscore_ / count
loss_precision_test_average = precision_ / count
loss_recall_test_average = recall_ / count
print('loss_cd_test_average:', loss_cd_test_average)
print('loss_fscore_test_average:', loss_fscore_test_average)
print('loss_precision_test_average:', loss_precision_test_average)
print('loss_recall_test_average:', loss_recall_test_average)

