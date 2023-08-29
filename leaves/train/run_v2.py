###############################################################
##### @Title:  birad cls baseline
##### @Time:  2022/7/22
##### @Author: frank
##### @Describe: 
        #  part0: data preprocess
        #  part1: build_transforme() & build_dataset() & build_dataloader()
        #  part2: build_model()
        #  part3: build_loss()
        #  part4: build_metric()
        #  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @To do: 
        #  data: multiresolution...
        #  model: resnxt, swin..
        #  loss: lovasz, HaussdorfLoss..
        #  infer: tta & pseudo labels...
##### @Reference:
###############################################################
import os
import pdb
import cv2
import time
import glob
import random

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
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            A.Rotate(limit=360, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.ColorJitter(p=0.2), # 在图像的颜色通道上引入随机扰动
            A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.ChannelShuffle(p=0.3), # 随机交换图像的颜色通道
                A.HueSaturationValue(p=0.3), # 用于调整图像的色调（Hue）、饱和度（Saturation）和亮度（Value）
            ], p=0.1),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2), # 随机生成粗糙的遮挡区域
            A.RandomShadow(p=0.2),
            # A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
            #                 min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
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
        self.train_val_flag = train_val_flag #
        self.img_paths = df['img_path'].tolist() 
        self.ids = df['img_name'].tolist()
        self.transforms = transforms
        
        if self.train_val_flag:
            self.label = df['img_label'].tolist()
        
    def __len__(self):
        return len(self.df)
        # return 128
    
    def __getitem__(self, index):
        
        id  = self.ids[index]
        img = Image.open(self.img_paths[index]).convert('RGB')
        
        if self.train_val_flag:
            data = self.transforms(image=np.array(img))
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            label = self.label[index]
            return torch.tensor(img), torch.from_numpy(np.array(label).astype(int))
        
        else:  # test
            ### augmentations
            data = self.transforms(image=np.array(img))
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            return torch.tensor(img), id

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
            # pdb.set_trace()
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
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device) 
        
        y_preds = model(images) 
        _, y_preds = torch.max(y_preds.data, dim=1)
        correct += (y_preds==gt).sum()
        
        total += gt.shape[0]
    
    val_acc = correct/total
    print("val_acc: {:.6f}".format(val_acc), flush=True)
    
    return val_acc

@torch.no_grad()
def test_one_epoch(ckpt_paths, test_loader, CFG):    

    pred_ids = []
    pred_cls = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids) in pbar:

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
            
        for id in ids:
            pred_ids.append(id)
    
    return pred_ids, pred_cls

def largest_factor(num):
    max_factor = 1
    i = 2
    
    while i <= num:
        if num % i == 0:
            max_factor = i
            num //= i
        else:
            i += 1
            
    return max_factor
if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # step2: data
        n_fold = 5
        img_size = [768, 768]
        train_bs = 32
        valid_bs = train_bs*2
        # step3: model
        backbone = 'efficientnet_b0'  
        # backbone = 'resnext50_32x4d'  
        num_classes = 3
        # step4: optimizer
        epoch = 20
        pseudo_epoch = 5
        lr = 1e-3
        wd = 1e-5
        lr_drop = 5
        # step5L infer
        TTA = True
        version = "v2"
        # step6: files
        ckpt_fold = "/home/zanzhuheng/Desktop/Working/leaves/leaves_ckpt"
        ckpt_name = "{}_{}_img{}_bs{}_fold{}_epoch{}".format(version,backbone,img_size[0],train_bs,n_fold,epoch)  # for submit. # 
        train_path = "/home/zanzhuheng/Desktop/Working/leaves/train"
        is_train = True
        is_test = False
        pseudo_lable = False
    
    set_seed(CFG.seed)
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = CFG.is_train
    if train_val_flag:
        ###############################################################
        ##### part0: data preprocess & simple EDA
        ###############################################################
        class_names = {'healthy':0, 'frog_eye_leaf_spot':1, 'scab':2}
        
        col_name = ['img_name', 'img_path', 'img_label']
        imgs_info = [] 
        for img_cls in os.listdir(CFG.train_path):
            for img_name in os.listdir(os.path.join(CFG.train_path, img_cls)):
                if img_name.endswith('.png'): # pass other files
                    imgs_info.append([img_name, os.path.join(CFG.train_path, img_cls, img_name), class_names[img_cls]])
    
        imgs_info_array = np.array(imgs_info)    
       
        df = pd.DataFrame(imgs_info_array, columns=col_name)
        
        # EDA by pandas: e.g. label freq 
        # df['img_label'].value_counts()
    
        ###############################################################
        ##### >>>>>>> trick1: cross validation train <<<<<<
        ###############################################################
        # document: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
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
            model = build_model(CFG, pretrain_flag=True) # init w/0 pre-train
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss

            best_val_acc = 0
            best_epoch = 0
            
            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_acc = valid_one_epoch(model, valid_loader, CFG)
                
                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = (val_acc > best_val_acc)
                best_val_acc = max(best_val_acc, val_acc)
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.6f}\n".format(epoch, epoch_time, best_val_acc), flush=True)


    test_flag = CFG.is_test
    if test_flag:
        ###############################################################
        ##### part0: data preprocess
        ###############################################################
        # real path: "/work/data/leafs-test-dataset"
        # fake path: "./leafs-test-dataset"
        
        test_path = "/work/data/leafs-test-dataset"
        import os
        files = os.listdir(test_path)   # 读入文件夹
        num_png = len(files)       # 统计文件夹中的文件个数
        CFG.valid_bs = largest_factor(num_png)
        print(CFG.valid_bs)
        col_name = ['img_name', 'img_path']
        imgs_info = [] 
        for img_name in os.listdir(test_path):
            if img_name.endswith('.png'): # pass other files
                imgs_info.append([img_name, os.path.join(test_path, img_name)])

        imgs_info_array = np.array(imgs_info)    
        df = pd.DataFrame(imgs_info_array, columns=col_name)

        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(df, train_val_flag=False, transforms=data_transforms['valid_test'])
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=False)

        ckpt_paths  = glob('./model/best*') # pick best ckpt for each fold.
        pred_ids, pred_cls = test_one_epoch(ckpt_paths, test_loader, CFG)
        
        
        class_names = ['healthy', 'frog_eye_leaf_spot', 'scab']
        pred_clsname=[]
        for i in pred_cls:
            pred_clsname.append(class_names[i])

        
        ###############################################################
        ##### step3: submit
        ###############################################################
        pred_df = pd.DataFrame({
            "uuid":pred_ids,
            "label": pred_clsname,
        })
        pred_df.to_csv('/work/output/result.csv',index=None)
    


    #!##############################################################
    #!#### >>>>>>>> Pseudo Labels <<<<<<<<
    #!##############################################################
    pseudo_label_flag = CFG.pseudo_lable
    if pseudo_label_flag:
        #!###########!replace###############replace############!replace#############!replace
        # test_path = "/work/data/birds-test-dataset"
        test_path = "/home/zanzhuheng/Desktop/Working/birds/pseudo_path"
        
        col_name = ['img_name', 'img_path']
        imgs_info = [] 
        for img_name in os.listdir(test_path):
            if img_name.endswith('.jpg'): # pass other files
                imgs_info.append([img_name, os.path.join(test_path, img_name)])

        imgs_info_array = np.array(imgs_info)    
        df = pd.DataFrame(imgs_info_array, columns=col_name)

        ###############################################################
        ##### >>>>>>> step2: infer <<<<<<
        ###############################################################
        data_transforms = build_transforms(CFG)  
        test_dataset = build_dataset(df, train_val_flag=False, transforms=data_transforms['valid_test'])
        test_loader  = DataLoader(test_dataset, batch_size=CFG.valid_bs, shuffle=False, pin_memory=False)

        ckpt_paths  = glob("./model/best*") # pick best ckpt for each fold.
        # pdb.set_trace()
        pred_ids, pred_cls = test_one_epoch(ckpt_paths, test_loader, CFG)
        df['img_label'] = pred_cls
        pesudo_df = df.copy()
        # pdb.set_trace()

        data_transforms = build_transforms(CFG)
        pseudo_dataset = build_dataset(pesudo_df, train_val_flag=True, transforms=data_transforms['valid_test'])#? augmentation or not?
        pseudo_loader  = DataLoader(pseudo_dataset, batch_size=CFG.valid_bs, shuffle=False, pin_memory=False)
        
        model = build_model(CFG, pretrain_flag=True) # init w/0 pre-train
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
        losses_dict = build_loss() # loss

        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(1, CFG.pseudo_epoch+1):
            train_one_epoch(model, pseudo_loader, optimizer, losses_dict, CFG)
            lr_scheduler.step()
            
            ###############################################################
            ##### >>>>>>> step4: save best model <<<<<<
            ###############################################################

            save_path = "/home/zanzhuheng/Desktop/Working/birds/my-inference-script-and-model/model/pseudo_model.pth"
            torch.save(model.state_dict(), save_path)
            
        ckpt_paths  =  glob("./model/*")
        # pdb.set_trace()
        pred_ids, pred_cls = test_one_epoch(ckpt_paths, test_loader, CFG)
        
        
        pred_df = pd.DataFrame({
            "imageID":pred_ids,
            "label": pred_cls,
        })
        #!###########!replace###############replace############!replace#############!replace
        # pred_df.to_csv('/work/output/result.csv',index=None)
        pred_df.to_csv('/home/zanzhuheng/Desktop/Working/birds/result.csv',index=None)

        