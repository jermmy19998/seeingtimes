
import os
import sys
import shutil
from joblib import Parallel, delayed
import multiprocessing
import copy
import argparse
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import glob
from easydict import EasyDict
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torchvision import transforms
from torchvision import models
from torch.utils.data._utils.collate import default_collate  # 导入默认的collate函数
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html


import cv2
from PIL import Image
import albumentations as A
from albumentations import RandomRotate90, Resize
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms as abtransforms

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
import timm



def set_seed(seed=1120):
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 训练早期停止机制
class EarlyStopping():
    def __init__(self, patience = 5,tol = 1e-3):
      self.patience = patience
      self.tol = tol
      self.counter = 0
      self.lowest_loss = None
      self.early_stop = False


    def __call__(self,val_loss):
       if self.lowest_loss is None:
          self.lowest_loss = val_loss
       elif self.lowest_loss - val_loss > self.tol:
            self.lowest_loss = val_loss
            self.counter = 0
       elif self.lowest_loss - val_loss < self.tol:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early Stopping Actived')
                self.early_stop = True
       return self.early_stop



# 混合精度训练
# https://zhuanlan.zhihu.com/p/408610877
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):

    model.train()
    scaler = amp.GradScaler() 
    correct = 0
    losses_all, ce_all, total  = 0, 0, 0 # 记录所有的损失
    path_error = []
    label_list = []

    pbar = tqdm(enumerate(train_loader), 
                total=len(train_loader), 
                desc='Train ')
    
    for _, (images, gt, path_imgs) in pbar:
        optimizer.zero_grad()
        label_list.extend(gt.tolist())
        images = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device)  

        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            losses = ce_loss
            _, y_preds = torch.max(y_preds.data, dim=1)
            correct += (y_preds==gt).sum()
            for j in range(len(gt)):
                cate_i = gt[j].cpu().numpy()
                pre_i = y_preds[j].cpu().numpy()
                if cate_i != pre_i:
                    path_error.append([cate_i, pre_i, path_imgs[j]])    # 记录错误样本的信息

        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item()
        ce_all += ce_loss.item()
        total += gt.shape[0]

    train_acc = correct/total   
    train_loss = losses_all/total
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all/total, ce_all/total, flush=True))

    return train_acc, train_loss, path_error
    
@torch.no_grad()
def valid_one_epoch(model, valid_loader, losses_dict, CFG):
    model.eval()
    correct = 0
    losses_all, ce_all, total  = 0, 0, 0 # 记录所有的损失
    path_error = []
    label_list = []
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid')
    for _, (images, gt, path_imgs) in pbar:
        label_list.extend(gt.tolist())
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device) 
        
        y_preds = model(images) 
        ce_loss = losses_dict["CELoss"](y_preds, gt.long())
        losses = ce_loss
        _, y_preds = torch.max(y_preds.data, dim=1)
        correct += (y_preds==gt).sum()
        for j in range(len(gt)):
                cate_i = gt[j].cpu().numpy()
                pre_i = y_preds[j].cpu().numpy()
                if cate_i != pre_i:
                    path_error.append([cate_i, pre_i, path_imgs[j]])    # 记录错误样本的信息
        
        losses_all += losses.item()
        ce_all += ce_loss.item()
        total += gt.shape[0]
    
    val_acc = correct/total
    val_loss = losses_all/total
    print("val_acc: {:.2f}".format(val_acc), flush=True)
    
    return val_acc.cpu().numpy(), val_loss.cpu().numpy(), path_error



@torch.no_grad()
def test_one_epoch(test_loader, CFG):
    # Load the model and weights
    model = build_model(CFG, pretrain_flag=False)  
    model.load_state_dict(torch.load(CFG.weight_path))
    model.eval()

    test_path = []
    test_probs = []
    test_preds = []
    test_trues = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test')
    for _, (images, gt, Path) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device) 
        logits = model(images) 

        test_probabilities, test_predictions = torch.max(F.softmax(logits, dim=1), 1) # 求出每一行的最大值
        test_probs.extend(test_probabilities.detach().cpu().numpy())
        test_preds.extend(test_predictions.detach().cpu().numpy())
        test_trues.extend(gt.detach().cpu().numpy())
        test_path.extend(Path)

    return test_path, test_probs, test_preds, test_trues


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir, checkpoint_type=None):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + f'_{checkpoint_type}.png'))
    plt.close()