
# -*-coding:utf-8 -*-
# ref: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
'''
@File    :	HE_MPR.py
@Time    :	2023/07/30 22:56:52
@Author  :	SeeingTimes
@Version :	1.0
@Contact :	wacto1998@gmail.com
@License :	ETH Zurich License


'''
import os
import pdb
import cv2
import time
import random
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate  # 导入默认的collate函数
from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
import albumentations as A # Augmentations
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report

def set_seed(seed=42):
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
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
        self.img_paths = df['Path'].tolist() 
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
            return torch.tensor(img), torch.from_numpy(np.array(label).astype(int)), self.img_paths[index]
        
        else:  # test
            ### augmentations
            data = self.transforms(image=np.array(img))
            img = np.transpose(data['image'], (2, 0, 1)) # [c, h, w]
            return torch.tensor(img), id

def build_dataloader(df, fold, data_transforms):
    train_data = df.query("fold!=@fold").reset_index(drop=True)
    valid_data = df.query("fold==@fold").reset_index(drop=True)
    # pdb.set_trace()
    train_dataset = build_dataset(train_data, train_val_flag=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_data, train_val_flag=True, transforms=data_transforms['valid_test'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=os.cpu_count(), shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader
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
    if os.path.exists(CFG.pretrained_ckpt_path):
        model.load_state_dict(torch.load(CFG.pretrained_ckpt_path))
    model.to(CFG.device)
    print('######Model activate --->{}<---######'.format(CFG.device),flush=True)
    return model



def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    return {"CELoss":CELoss}


# 混合精度训练
# https://zhuanlan.zhihu.com/p/408610877
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler() 
    losses_all, ce_all, total  = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _,  (images, gt,path) in pbar:
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
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all/total, ce_all/total, flush=True))
    
@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt,path) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device) 
        
        y_preds = model(images) 
        _, y_preds = torch.max(y_preds.data, dim=1)
        correct += (y_preds==gt).sum()
        
        total += gt.shape[0]
    
    val_acc = correct/total
    print("val_acc: {:.2f}".format(val_acc), flush=True)
    
    return val_acc


@torch.no_grad()
def test_one_epoch(test_loader, CFG, weight_path):
    # Load the model and weights
    model = build_model(CFG, pretrain_flag=False)  
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    test_path = []
    test_probs = []
    test_preds = []
    test_trues = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test ')
    for _, (images, gt,path) in pbar:
        images  = images.to(CFG.device, dtype=torch.float) # [b, c, w, h]
        gt  = gt.to(CFG.device) 
        logits = model(images) 

        test_probabilities, test_predictions = torch.max(F.softmax(logits, dim=1), 1) # 求出每一行的最大值
        test_probs.extend(test_probabilities.detach().cpu().numpy())
        test_preds.extend(test_predictions.detach().cpu().numpy())
        test_trues.extend(gt.detach().cpu().numpy())
        test_path.extend(path)

    return test_path, test_probs, test_preds, test_trues





@torch.no_grad()
def infer_one_epoch(image_path, weight_path):
    # Load the model and weights
    model = build_model(CFG, pretrain_flag=False)  
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    transform = data_transforms["valid_test"]

    # Load and preprocess the image (PIL图像转换为NumPy数组)
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # 将PIL图像转换为NumPy数组
    img = transform(image=img)["image"]
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    img_tensor = img_tensor.to(CFG.device)

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)

    # Process the output (adjust as per your model's output format)
    predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class


# 可视化函数
def visualize_prediction(image_path, true_label, predicted_label):
    # Load and show the image
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.title(f"Ground Truth: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()



if __name__ == '__main__':

    class CFG:
        Wsi_id = "Patient_id"
        # hyper-parameter
        seed = 42 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # data
        n_fold = 5
        img_size = [1024, 1024]
        train_bs = 15
        valid_bs = train_bs * 2 
        # step3: model
        backbone = 'efficientnet_b0'  
        num_classes = 2
        # optimizer
        epoch = 50
        lr = 1e-3
        wd = 1e-5
        lr_drop = 5
        ckpt_fold = "HE_MPR Checkpoint"
        ckpt_name = "efficientnetb0_img1024_bs32_fold5_epoch30"  
        pretrained_ckpt_path = r"C:\Users\Administrator\bin\huangzongyao\torchkit\classification\tools\HE_MPR Checkpoint\efficientnetb0_img1024_bs32_fold5_epoch30\best_fold4_epoch5.pth"
        
    
    # 按照集合(train、test、valid)来分割数据
    df = pd.read_csv(r'E:\zhouyehan\single_label_Proj\HE_df_all_1024_filter.csv') #! Your csv file path

    # train_data = df[df['Set'] == 'train']
    # test_data = df[df['Set'] == 'test']
    # valid_data = df[df['Set'] == 'valid']

    # pd.crosstab(df['id'], df['Set'])
    
    set_seed(CFG.seed)
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = False
    if train_val_flag:

                

        # document: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
        # kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        # for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        #     df.loc[val_idx, 'fold'] = fold

        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df[CFG.Wsi_id].unique())):
            val_wsi_id = df[CFG.Wsi_id].unique() [val_idx]
            df.loc[df[CFG.Wsi_id].isin(val_wsi_id.tolist()), 'fold'] = fold

        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

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
                train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_acc = valid_one_epoch(model, valid_loader, CFG)
                is_best = (val_acc > best_val_acc)
                best_val_acc = max(best_val_acc, val_acc)
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path) 
                    torch.save(model.state_dict(), save_path)
                
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_acc), flush=True)


    test_flag = True
    if test_flag:
        print(f'#'*40, flush=True)
        print(f'###### Infer', flush=True)
        print(f'#'*40, flush=True)
        test_data = pd.read_csv(r"E:\zhouyehan\single_label_Proj\TEST_patch\test_df_all.csv")
        data_transforms = build_transforms(CFG)
        test_dataset = build_dataset(test_data, train_val_flag=True, transforms=data_transforms['valid_test'])
        test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)

        ckpt_paths = glob(r'C:\Users\Administrator\bin\huangzongyao\torchkit\classification\tools\HE_MPR Checkpoint\efficientnetb0_img1024_bs32_fold5_epoch30\best_fold4_epoch5.pth') #! 替换做好的pth文件路径

        # image_path = valid_data['Path'].tolist()[0]
        # true_label = valid_data['Label'].tolist()
        weight_path = ckpt_paths[0]

        # # 进行预测
        # predicted_label = infer_one_epoch(image_path, weight_path)

        # # 可视化图像、Ground Truth和预测结果
        # visualize_prediction(image_path, true_label, predicted_label)

        test_path, test_probs, test_preds, test_trues = test_one_epoch(test_loader, CFG, weight_path)
        test_log = pd.DataFrame({"path":test_path,"probs":test_probs,"preds":test_preds,"gt":test_trues})
        test_log.insert(loc = 0,column = "pred_score_1", value = list(map(lambda x: x[0] if x[1] == 1 else 1-x[0], np.array(test_log[['probs', 'preds']]))))
        test_log.to_csv(r'E:\zhouyehan\single_label_Proj\TEST_patch\test_log.csv')
        accuracy = accuracy_score(test_preds, test_trues)
        weighted_f1_score = f1_score(test_preds, test_trues, average='weighted')
        report = classification_report(test_preds, test_trues, digits=4)
        print('Test weighted F1 score {}'.format(weighted_f1_score))
        print('Test accuracy {}'.format(accuracy))
        print('Test classification report {}'.format(report))

