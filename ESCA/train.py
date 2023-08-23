import os
import sys
sys.path.append("/mnt/raid/Archive/Public/WSI/svs/jijuan/Breast_proj")
import pdb
import cv2
import time
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
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate  # 导入默认的collate函数
from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
from sklearn.metrics import accuracy_score, f1_score, classification_report
import albumentations as A # Augmentations
import timm
import copy
import gc

from config_relate import CFG
from dataloader import build_transforms, build_dataset, build_dataloader
from utils import build_model, build_loss, train_one_epoch, valid_one_epoch, test_one_epoch, EarlyStopping, plot_line



if CFG.train_mode :
    # document: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(CFG.df[CFG.Wsi_id].unique())):
            val_wsi_id = CFG.df[CFG.Wsi_id].unique() [val_idx]
            CFG.df.loc[CFG.df[CFG.Wsi_id].isin(val_wsi_id.tolist()), 'fold'] = fold

        for fold in range(CFG.n_fold):
            print(f'#'*40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#'*40, flush=True)

            data_transforms = build_transforms(CFG)  
            train_loader, valid_loader = build_dataloader(CFG, fold, data_transforms)
            model = build_model(CFG, pretrain_flag=True) # init w/0 pre-train
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop) 
            losses_dict = build_loss() # loss

            best_model_wts = copy.deepcopy(model.state_dict())
            best_index = {'acc': 0.0,"epoch":1}
            early_stoping = EarlyStopping(patience=CFG.patience,tol=CFG.tol) # 实例化提前终止类
            loss_rec = {"train": [], "valid": []} # 记录每一个epoch的平均loss结果
            acc_rec = {"train": [], "valid": []} # 记录每一个epoch的平均acc结果
            error_info = {"train": [], "valid": []}


            for epoch in range(1, CFG.epoch+1):
                start_time = time.time()
                train_acc, train_loss, path_error_train = train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_acc, val_loss, path_error_valid = valid_one_epoch(model, valid_loader, losses_dict, CFG)

                # 记录训练信息
                loss_rec["train"].append(train_loss), loss_rec["valid"].append(val_loss)
                acc_rec["train"].append(train_acc), acc_rec["valid"].append(val_acc)

                # 保存loss曲线， acc曲线
                plt_x = np.arange(1, epoch + 1)
                plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir= CFG.ckpt_path, checkpoint_type=f"Train_fold{fold}")
                plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir= CFG.ckpt_path, checkpoint_type=f"Train_fold{fold}")

                if val_acc > best_index['acc']:
                    best_index['acc'] = val_acc
                    best_index['epoch'] = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_index['acc']), flush=True)

                error_info["train"].append(pd.DataFrame(path_error_train,columns=["gt","pred","img_path"]))
                error_info["valid"].append(pd.DataFrame(path_error_valid,columns=["gt","pred","img_path"]))
                

                # 提前终止 判断
                if early_stoping(val_loss):
                    break

                # 释放每一个epoch的内存
                del train_acc, train_loss, val_acc, val_loss
                gc.collect()
                torch.cuda.empty_cache()

            error_info_train = pd.concat(error_info["train"],axis=0)
            error_info_train.insert(0,"type","train")
            error_info_valid = pd.concat(error_info["valid"],axis=0)
            error_info_valid.insert(0,"type","valid")
            error_info = pd.concat([error_info_train,error_info_valid],axis=0)
            error_info.to_csv(f"{CFG.ckpt_path}/error_info_fold{fold}.csv",index=False)
            
            save_path = f"{CFG.ckpt_path}/best_fold{fold}_epoch{best_index['epoch']}_{best_index['acc']}.pth" 
            torch.save(best_model_wts, save_path)

if CFG.test_mode:
     data_transforms = build_transforms(CFG)
     test_dataset = build_dataset(CFG.test_data, CFG, train_val_flag=True, transforms=data_transforms['valid_test'])
     test_loader = DataLoader(
                test_dataset,
                batch_size=CFG.valid_bs,
                num_workers=10,
                shuffle=False,
                pin_memory=False,
                collate_fn=default_collate  # 使用正确的collate函数
            )
     test_path, test_probs, test_preds, test_trues = test_one_epoch(test_loader, CFG)
     test_log = pd.DataFrame({"img_path":test_path,"probs":test_probs,"preds":test_preds,"gt":test_trues})
     test_log.insert(loc = 1,column = "pred_score_1", value = list(map(lambda x: x[0] if x[1] == 1 else 1-x[0], np.array(test_log[['probs', 'preds']]))))
     test_log.to_csv(CFG.test_log_path)
     accuracy = accuracy_score(test_preds, test_trues)
     weighted_f1_score = f1_score(test_preds, test_trues, average='weighted')
     report = classification_report(test_preds, test_trues, digits=4)
     print('Test weighted F1 score {}'.format(weighted_f1_score))
     print('Test accuracy {}'.format(accuracy))
     print('Test classification report {}'.format(report))
     
