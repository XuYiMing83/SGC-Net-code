import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pytorch3d.loss
from pytorch3d.ops import sample_farthest_points
import itertools
from Model.Dataset import PointsDataSet
#from pointmodel.modelcoarse import FeatureDecoder_coarse, Loss
#from smodel.model9 import GRNet
from Model.SGC_model import SGC, Loss
#from pointmodel.model import FeatureDecoder
from tqdm import tqdm
import open3d as o3d
import logging
from pathlib import Path
import datetime
import time
import random
import wandb
#'''
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
seed = 20222022
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

wandb.init(
    # Set the project where this run will be logged
    project="SGC02end_no_MLP",
    dir='E:/wandb/',
    # Track hyperparameters and run metadata
    config={
        "learning_rate": '0.001, ExponentialLR(gamma=0.8)',
        "architecture": "MLP",
        "epochs": 0,
    })

all_root = 'D:/MA/D_MA/27648/data_all'
print('loading data ...')
train_dataset = PointsDataSet(root=all_root, gt='sub_gt', training='sub_training')
# train_dataset = PointsDataSet(root=all_root, gt='sub_test_gt', training='sub_test_training')
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

test_dataset = PointsDataSet(root=all_root, gt='sub_test_gt', training='sub_test_training')
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

save_number = 'SGC02end_no_MLP'
save_gt_root = 'D:/MA/D_MA/27648/sres_gt{}'.format(save_number)
save_test_root = 'D:/MA/D_MA/27648/stest_result{}'.format(save_number)
save_pth_root = 'E:/pth_s{}/'.format(save_number)
if not os.path.exists(save_gt_root):
    os.makedirs(save_gt_root)
if not os.path.exists(save_test_root):
    os.makedirs(save_test_root)
if not os.path.exists(save_pth_root):
    os.makedirs(save_pth_root)

timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
experiment_dir = Path('./log/')
experiment_dir.mkdir(exist_ok=True)
experiment_dir = experiment_dir.joinpath(timestr)
experiment_dir.mkdir(exist_ok=True)

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
file_handler = logging.FileHandler(f"{experiment_dir}/logs.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
def log_string(str):
    logger.info(str)
    print(str)
#'''
model_path = 'E:/model/model_cube.pth'   # 'E:/pth_s16/model_best_refine.pth'
model_loader = torch.load(model_path, map_location='cpu')

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

CNN3D_dflt = 40
convn_list = list([CNN3D_dflt, CNN3D_dflt * 2, CNN3D_dflt * 4, CNN3D_dflt * 8, CNN3D_dflt * 8 * 5 * 5 * 5])

Model = SGC(convlist=convn_list)
print(get_parameter_number(Model))
Model_dict = Model.state_dict()
pretrained_dict = model_loader#['model_state_dict']
key_list = list()
for k, v in model_loader.items():
    key_list.append(k)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in Model_dict}
Model_dict.update(pretrained_dict)
Model.load_state_dict(Model_dict)
for k, v in Model.named_parameters():
    if k in key_list:
        v.requires_grad = False
# Model.load_state_dict(model_loader['model_state_dict'])
for k, v in Model.named_parameters():
    print(k, v.requires_grad)

# Model.load_state_dict(model_loader)
'''
for param_tensor in pretrained_dict:
    print(param_tensor, '\t', pretrained_dict[param_tensor])
print('#########################################################')
for param_tensor in Model.state_dict():
    print(param_tensor, '\t', Model.state_dict()[param_tensor])
print('#########################################################')
raise NotImplementedError()
'''
Model = Model.cuda()
#print(sum(p.numel() for p in Model.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(Model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08,)
#optimizer = Ranger(Model.parameters(), lr=0.0001)
#optimizer.load_state_dict(model_loader['optimizer_state_dict'])
scalar = 0.95
#criterion = Loss().cuda()
criterion = Loss().cuda()
global_epoch = 0#model_loader['epoch'] + 1
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97, last_epoch=global_epoch-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120, 140, 160], gamma=0.6)
best_loss = 1000000#model_loader['loss:']
best_loss_fine = 1000000#model_loader['loss_fine:']
for epoch in range(global_epoch, 271):
    log_string('Epoch start_epoch:{} epoch:{}:'.format(global_epoch, epoch))
    log_string('Learning rate:{}'.format(get_learning_rate(optimizer)))
    Batch = 1
    s = time.time()
    alpha = 1

    for data in tqdm(train_dl, total=len(train_dl), smoothing=0.9):
        #for data in tqdm(train_dl, total=len(train_dl), smoothing=0.9):
        training, gt, _ = data # B, 3, N
        training = training.cuda()
        training = training.float()
        gt = gt.cuda()
        gt = gt.float()

        optimizer.zero_grad()
        Model = Model.train()

        coarse, fine = Model(training)
        loss, loss_coarse, loss_fine = criterion(coarse, fine, gt, alpha=alpha)

        #log_string('epoch:{}: batch:{}'.format(epoch, Batch))
        #log_string('loss:{}'.format(loss))
        wandb.log({'epoch：': epoch, 'batch:':Batch, 'loss:': loss,
                       'coarse_loss:': loss_coarse, 'fine_loss:': loss_fine})
        loss.backward()
        optimizer.step()
        Batch += 1


    log_string('epoch:{} loss:{}'.format(epoch, loss))
    log_string('epoch time:{}s'.format(time.time() - s))
    scheduler.step()


    with torch.no_grad():
        test_result = torch.zeros(len(test_dl)).cuda()
        test_result_fine = torch.zeros(len(test_dl)).cuda()
        test_result_refine = torch.zeros(len(test_dl)).cuda()
        for batch_id, (training_, gt_, _) in tqdm(enumerate(test_dl), total=len(test_dl), smoothing=0.9):
            training_ = training_.cuda()
            training_ = training_.float()
            gt_ = gt_.cuda()
            gt_ = gt_.float()

            Model = Model.eval()
            coarse_, fine_ = Model(training_)
            loss_, loss_coarse_, loss_fine_ = criterion(coarse_, fine_, gt_, alpha=alpha)
            test_result[batch_id] = loss_
            test_result_fine[batch_id] = loss_coarse_
            test_result_refine[batch_id] = loss_fine_
            #break

        mean_loss = test_result.mean()
        mean_coarse = test_result_fine.mean()
        mean_fine = test_result_refine.mean()
        log_string(
            'epoch_test:{} loss:{}'.format(epoch, mean_loss))
        wandb.log({'Test loss:': mean_loss, 'Test coarse loss:': mean_coarse, 'Test fine loss:': mean_fine})


        a = np.array(gt_.transpose(2, 1).cpu())[0]/scalar
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(a)
        o3d.io.write_point_cloud(os.path.join(save_test_root, '{}_gt.ply'.format(epoch)), pcd)

        b = fine_.cpu().detach().numpy()[0]/scalar
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(b)
        o3d.io.write_point_cloud(os.path.join(save_test_root, '{}_fine.ply'.format(epoch)),
                                 pcd2)


        d = coarse_.cpu().detach().numpy()[0]/scalar
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(d)
        o3d.io.write_point_cloud(os.path.join(save_test_root, '{}_coarse.ply'.format(epoch)), pcd4)

        e = np.array(training_.transpose(2, 1).cpu())[0]/scalar
        pcd5 = o3d.geometry.PointCloud()
        pcd5.points = o3d.utility.Vector3dVector(e)
        o3d.io.write_point_cloud(os.path.join(save_test_root, '{}_training.ply'.format(epoch)), pcd5)



        #
        a2 = np.array(gt.transpose(2, 1).cpu())[0]/scalar
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(a2)
        o3d.io.write_point_cloud(os.path.join(save_gt_root, '{}_gt.ply'.format(epoch)), pcd)

        b2 = fine.cpu().detach().numpy()[0]/scalar
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(b2)
        o3d.io.write_point_cloud(os.path.join(save_gt_root, '{}_fine.ply'.format(epoch)),
                                 pcd2)

        d2 = coarse.cpu().detach().numpy()[0]/scalar
        pcd4 = o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(d2)
        o3d.io.write_point_cloud(os.path.join(save_gt_root, '{}_coarse.ply'.format(epoch)), pcd4)

        e2 = np.array(training.transpose(2, 1).cpu())[0]/scalar
        pcd5 = o3d.geometry.PointCloud()
        pcd5.points = o3d.utility.Vector3dVector(e2)
        o3d.io.write_point_cloud(os.path.join(save_gt_root, '{}_training.ply'.format(epoch)), pcd5)
        #break


        if epoch % 15 == 0:
            logger.info('Save model...')
            savepath = os.path.join(save_pth_root, 'model_{}.pth'.format(epoch))
            log_string('Saving at %s' % savepath)
            state = {'epoch': epoch,
                     'loss:': mean_loss,
                     'loss_fine:': mean_coarse,
                     'loss_refine:': mean_fine,
                     'model_state_dict': Model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, savepath)
            log_string('Saving model....')
        if mean_fine < best_loss_fine:
            log_string('best_loss_fine:{}'.format(mean_fine))
            best_loss_fine = mean_fine
            logger.info('Save model best fine...')
            savepath4 = os.path.join(save_pth_root, 'model_best_fine.pth')
            log_string('Saving at %s' % savepath4)
            state = {'epoch': epoch,
                     'loss:': mean_loss,
                     'loss_fine:': mean_coarse,
                     'loss_refine:': mean_fine,
                     'model_state_dict': Model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, savepath4)
            log_string('Saving model best fine....')
        if mean_loss < best_loss:
            log_string('best_loss_loss:{}'.format(mean_loss))
            best_loss = mean_loss
            logger.info('Save model best loss...')
            savepath4 = os.path.join(save_pth_root, 'model_best_loss.pth')
            log_string('Saving at %s' % savepath4)
            state = {'epoch': epoch,
                     'loss:': mean_loss,
                     'loss_fine:': mean_coarse,
                     'loss_refine:': mean_fine,
                     'model_state_dict': Model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, savepath4)
            log_string('Saving model best fine....')
wandb.finish()