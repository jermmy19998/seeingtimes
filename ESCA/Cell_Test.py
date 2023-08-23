#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :	HE.py
@Time    :	2023/08/14 21:49:06
@Author  :	SeeingTimes
@Version :	1.0
@Contact :	wacto1998@gmail.com
@License :	ETH Zurich License


'''
import os
import pdb
import cv2
import time
import glob
import random
from datetime import datetime

from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html
import torch.nn.functional as F

from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
import albumentations as A # Augmentations
import timm
# import segmentation_models_pytorch as smp # smp

def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################
##### part0: data preprocess
###############################################################

###############################################################
##### part1: build_transforms & build_dataset & build_dataloader
###############################################################
# document: https://albumentations.ai/docs/
# example: https://github.com/albumentations-team/albumentations_examples
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            # # dimension should be multiples of 32.
            # ref: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
            A.CLAHE(clip_limit=(1,10), p= 1), # before normalize
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.8),
            A.RandomRotate90(),
            A.RandomCrop(*CFG.img_size, p=0.55),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7), 
            A.Transpose(),
            # A.CLAHE(clip_limit=(1,10), p= 1),
            A.Rotate(limit=90, p=1),
            # A.RandomBrightnessContrast(p=0.8), # for test
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.8),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                A.Blur(blur_limit=3),
            ], p=0.1),
            ]),
        
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
            ])
        }
    return data_transforms

class build_dataset(Dataset):
    def __init__(self, df, train_val_flag=True, transforms=None):
        
        self.df = df
        # self.id = df['filename']
        self.train_val_flag = train_val_flag #
        self.img_paths = df['Path'].tolist() 
        # pdb.set_trace()
        self.transforms = transforms

        
        if self.train_val_flag:
            self.label = df['Label'].tolist()
            
        
    def __len__(self):
        return len(self.df)
        
    
    def __getitem__(self, index):
        
        img = Image.open(self.img_paths[index]).convert('RGB')
        
        if self.train_val_flag:
            data = self.transforms(image=np.array(img))
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            label = self.label[index]
            # pdb.set_trace()
            return torch.tensor(img), torch.from_numpy(np.array(label).astype(int))
        
        else:  # test
            ### augmentations
            data = self.transforms(image=np.array(img))
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            return torch.tensor(img)

def build_dataloader(df, fold, data_transforms):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = build_dataset(train_df, train_val_flag=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_df, train_val_flag=True, transforms=data_transforms['valid_test'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=os.cpu_count(), shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://timm.fast.ai/create_model
# cls model repos: timm.list_models()
def build_model(CFG, pretrain_flag=False):
    if pretrain_flag:
        pretrain_weights = "imagenet"
    else:
        pretrain_weights = False
    model = timm.create_model(CFG.backbone, 
                              pretrained=pretrain_weights, 
                              num_classes=CFG.num_classes)
    model.to(CFG.device)
    return model

###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
# def build_loss():
#     BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
#     TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
#     return {"BCELoss":BCELoss, "TverskyLoss":TverskyLoss}

def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    return {"CELoss":CELoss}

# TODO Math
# def build_loss():
#     SL1Loss = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')
#     return {"SL1Loss":SL1Loss}


###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
    
###############################################################
##### >>>>>>> part5: train & validation & test <<<<<<
###############################################################
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, ce_all, total  = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, gt) in pbar:
        optimizer.zero_grad()

        images = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device)  

        with amp.autocast(enabled=True):
            y_preds = model(images) # [b, c, w, h]
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            losses = ce_loss
        
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses_all += losses.item()
        ce_all += ce_loss.item()
        total += gt.shape[0]
        
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.6f}".format(current_lr), flush=True)
    print("loss: {:.6f}, ce_all: {:.6f}".format(losses_all/total, ce_all/total, flush=True))
    
@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    all_preds = []
    all_gts = []
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt) in pbar:
        images = images.to(CFG.device, dtype=torch.float)
        gt = gt.to(CFG.device)
        
        y_preds = model(images) 
        _, y_preds = torch.max(y_preds.data, dim=1)
        
        all_preds.extend(y_preds.cpu().numpy())
        all_gts.extend(gt.cpu().numpy())
    
    val_acc = (np.array(all_preds) == np.array(all_gts)).mean()
    val_f1 = f1_score(all_gts, all_preds, average='binary')
    val_precision = precision_score(all_gts, all_preds, average='binary')
    val_recall = recall_score(all_gts, all_preds, average='binary')

    print("val_acc: {:.6f}, F1 Score: {:.6f}, Precision: {:.6f}, Recall: {:.6f}".format(
        val_acc, val_f1, val_precision, val_recall), flush=True)
    
    return {
        'accuracy': val_acc,
        'f1_score': val_f1,
        'precision': val_precision,
        'recall': val_recall
    }

@torch.no_grad()
def test_one_epoch(ckpt_paths, test_loader, CFG):    

    pred_ids = []
    pred_cls = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images) in pbar:

        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
    
        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################
        y_prob_vote = torch.zeros((CFG.valid_bs, CFG.num_classes), device=CFG.device, dtype=torch.float32) # [bs, num_cls]
        
        for sub_ckpt_path in ckpt_paths:
            model = build_model(CFG, pretrain_flag=False)
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            
            y_preds = model(images) # [bs, num_cls] 
            y_prob = F.softmax(y_preds, dim=1)

            # pdb.set_trace()
            y_prob_vote += y_prob
            
            # ############################################
            # ##### >>>>>>> TTA  <<<<<<
            # ############################################
            if CFG.TTA:
                images_f = [*[torch.rot90(images, k=i, dims=(-2, -1)) for i in range(1, 4)]]
                for image_f in images_f:
                     y_preds_f = model(image_f)
                     y_prob_f = F.softmax(y_preds_f, dim=1) # just for cls
                     y_prob_vote += y_prob_f
        
        num_vote = CFG.n_fold
        if CFG.TTA:
            num_vote = CFG.n_fold * 3

        y_prob_vote /= num_vote
        cls_pred = y_prob_vote.argmax(1)

        for pred in cls_pred.data.cpu().numpy():
            pred_cls.append(pred)
            
        # for id in ids:
        #     pred_ids.append(id)
    
    return pred_ids, pred_cls


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # step2: data
        n_fold = 5
        img_size = [512, 512]
        train_bs = 32
        valid_bs =  4# Test dataset长度的最大公约数 for HE
        # valid_bs = 42 # for CD
        # model
        backbone = 'efficientnet_b1'  
        num_classes = 2
        # optimizer
        epoch = 15
        lr = 1e-3
        wd = 1e-5
        lr_drop = 8
        version = "v3_PD"
        ckpt_fold = "PD Checkpoint"
        ckpt_name = "{}_{}_img{}_bs{}_fold{}_epoch{}".format(version,backbone,img_size[0],train_bs,n_fold,epoch)  # for submit. # 
        TTA = True # test time augmentation
        is_Train = False
        is_Test = True
    
    set_seed(CFG.seed)
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = CFG.is_Train
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess & simple EDA
        ###############################################################
        class_names = {0,1}
        
        df = pd.read_csv('/home/zanzhuheng/Desktop/Working/ESCA/ESCA_pre_train_out_230810/PD/PD_balanced_1024_filter_train_val.csv')
        
        # EDA by pandas: e.g. label freq 
        # df['img_label'].value_counts()
    
        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################
        # document: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
        total_start = time.time()
        # 获取当前时间戳

        # 将时间戳转换为日期和时间格式
        date_time = datetime.fromtimestamp(total_start).strftime('%Y-%m-%d %H:%M:%S')
        print(" Start Time at : {}; Unix : {}".format(date_time,total_start),flush=True)
        
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold

        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            ###############################################################
            ##### >>>>>>> step2: combination <<<<<<
            ##### build_transforme() & build_dataset() & build_dataloader()
            ##### build_model() & build_loss()
            ###############################################################
            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms)
            model = build_model(CFG, pretrain_flag=False) # init w/0 pre-train
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss

            best_val_acc = 0
            best_epoch = 0
            best_val_f1 = 0
            
        
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_metrics = valid_one_epoch(model, valid_loader, CFG)
                val_acc = val_metrics['accuracy']
                val_f1 = val_metrics['f1_score']
                
                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = ((val_acc > best_val_acc) or (val_f1 > best_val_f1))
                best_val_acc = max(best_val_acc, val_acc)
                best_val_f1 = max(best_val_f1, val_f1)
                
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                total_end = time.time() - total_start
                print("epoch:{}, time:{:.2f}s, best_acc:{:.6f}, best best_val_f1:{:6f}\n".format(epoch, epoch_time, best_val_acc,best_val_f1), flush=True)
            print("End Of Train, Total cost time :{:.2f}".format(total_end), flush=True)


    test_flag = CFG.is_Test
    if test_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        # real path: "/work/data/birds-test-dataset"
        # fake path: "./birds-test-dataset"
        

        df = pd.read_csv('/home/zanzhuheng/Desktop/Working/ESCA/ESCA_pre_train_out_230810/PD/PD_balanced_test.csv')

        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(df, train_val_flag=False, transforms=data_transforms['valid_test'])
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=os.cpu_count(), shuffle=False, pin_memory=False)

        ckpt_paths  = glob('/home/zanzhuheng/PD Checkpoint/v3_PD_efficientnet_b1_img512_bs8_fold5_epoch15/best_fold4_epoch13.pth') # pick best ckpt for each fold.

        # ckpt_paths = [path for path in ckpt_paths if "best_fold3*" in path or 'best_fold4*' in path]
        pred_ids, pred_cls = test_one_epoch(ckpt_paths, test_loader, CFG)

        ###############################################################
        ##### step3: submit
        ###############################################################
        pred_df = pd.DataFrame({
            "imageID": df['Path'],
            "predict_label": pred_cls,
            "ground truth": df['Label']
        })

        correct_predictions = (pred_cls == df['Label']).sum()
        total_samples = df.shape[0]

        accuracy = correct_predictions / total_samples
        print("Accuracy:", round(accuracy,2),flush=True)

        f1 = f1_score(df['Label'], pred_cls, average='weighted')  # Calculate weighted F1 score
        print("F1 Score:", round(f1,2),flush=True)

        pred_df.to_csv('/home/zanzhuheng/Desktop/Working/ESCA/result_{}.csv'.format(CFG.ckpt_name),index=None)

        