import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import utils
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from utils import SinScheduler, GaussianRampUpScheduler
from tqdm import tqdm, trange
from inference import infer_once
import random
import torch
import numpy as np
from dataset import FullLabelDataset, collate_fn, NoLabelDataset
from model import FullModel, EMA
from dataloader import BinaryDataLoader

import yaml
from argparse import Namespace

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=utils.dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    vocab = yaml.safe_load(file)

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)
random.seed(config.random_seed)

if __name__ == '__main__':

    full_train_dataset = FullLabelDataset(name='train')
    full_train_dataloader = DataLoader(dataset=full_train_dataset, batch_size=config.batch_size_sup, shuffle=True,collate_fn=collate_fn)
    full_train_dataiter = iter(full_train_dataloader)

    # usp_dataset = NoLabelDataset(name='train')
    # usp_dataloader = DataLoader(dataset=usp_dataset, batch_size=config.batch_size_usp, shuffle=True,collate_fn=collate_fn)
    # usp_dataiter = iter(usp_dataloader)

    usp_scheduler = GaussianRampUpScheduler(config.max_steps,0,config.max_steps)

    valid_dataset = FullLabelDataset(name='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size_sup, shuffle=False,collate_fn=collate_fn,drop_last=True)

    model=FullModel().to(config.device)
    # ema = EMA(model, 0.99)
    # ema.register()
    # CE_loss_fn=nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    EMD_loss_fn=utils.BinaryEMDLoss()
    # BCE_loss_fn=nn.BCELoss()
    MSE_loss_fn=nn.MSELoss()
    seg_GHM_loss_fn=utils.GHMLoss(vocab['<vocab_size>'],num_prob_bins=10,alpha=0.999,label_smoothing=config.label_smoothing)
    edge_GHM_loss_fn=utils.GHMLoss(2,num_prob_bins=5,alpha=0.99999,label_smoothing=config.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=config.max_steps)

    progress_bar = tqdm(total=config.max_steps, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    model_name='model'
    writer = SummaryWriter()
    print('start training')
    for i in range(config.max_steps):
        model.train()
        optimizer.zero_grad()

        # full supervised training
        try:
            melspec,target,edge_target=next(full_train_dataiter)
        except StopIteration:
            full_train_dataiter = iter(full_train_dataloader)
            melspec,target,edge_target=next(full_train_dataiter)

        melspec,target,edge_target=\
        torch.tensor(melspec).to(config.device),\
        torch.tensor(target).to(config.device).long().squeeze(1),\
        torch.tensor(edge_target).to(config.device).long()
        h,seg,ctc,edge=model(melspec)
        
        is_edge_prob=F.softmax(edge,dim=1)[:,0,:]
        seg_loss=seg_GHM_loss_fn(seg,target.squeeze(1))
        edge_loss=edge_GHM_loss_fn(edge,edge_target.squeeze(1))+EMD_loss_fn(is_edge_prob,edge_target)
        loss=edge_loss+seg_loss

        writer.add_scalar('Loss/train/Accuracy', (seg.argmax(dim=1)==target).float().mean().item(), i)
        writer.add_scalar('Loss/train/sup/seq', seg_loss.item(), i) 
        writer.add_scalar('Loss/train/sup/edge', edge_loss.item(), i)

        # semi supervised training
        if usp_scheduler()>0:
            pass
            # try:
            #     feature, feature_weak_aug, feature_strong_aug=next(usp_dataiter)
            # except StopIteration:
            #     usp_dataiter = iter(usp_dataloader)
            #     feature, feature_weak_aug, feature_strong_aug=next(usp_dataiter)

            # feature, feature_weak_aug, feature_strong_aug=feature.to(config.device), feature_weak_aug.to(config.device), feature_strong_aug.to(config.device)
            # h,seg,ctc,edge=model(feature)
            # h_weak,seg_weak,ctc_weak,edge_weak=model(feature_weak_aug)
            # h_strong,seg_strong,ctc_strong,edge_strong=model(feature_strong_aug)
            # consistence_loss=(
            #     MSE_loss_fn(seg_weak,seg)+MSE_loss_fn(seg_strong,seg)+MSE_loss_fn(seg_strong,seg_weak)+\
            #     MSE_loss_fn(edge_weak,edge)+MSE_loss_fn(edge_strong,edge)+MSE_loss_fn(edge_strong,edge_weak)
            # )

            # writer.add_scalar('Loss/train/consistence', consistence_loss.item(), i)

            # loss+=usp_scheduler()*consistence_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        usp_scheduler.step()
        # ema.update()

        writer.add_scalar('Loss/train/total', loss.item(), i)
        writer.add_scalar('learning_rate/total', optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('learning_rate/usp', usp_scheduler(), i)
        progress_bar.set_description(f'tr_loss: {loss.item():.3f}')
        progress_bar.update()

        if i%config.val_interval==0:
            # pass
            # print('validating...')
            model.eval()
            # ema.apply_shadow()
            y_true=[]
            y_pred=[]
            val_loss_seg=[]
            val_loss_edge=[]
            with torch.no_grad():
                for melspec,target,edge_target in valid_dataloader:
                    melspec,target,edge_target=melspec.to(config.device),target.to(config.device).squeeze(1),edge_target.to(config.device).squeeze(1).long()
                    h,seg,ctc,edge=model(melspec)

                    is_edge_prob=F.softmax(edge,dim=1)[:,0,:]

                    y_true.append(target.cpu())
                    y_pred.append(seg.argmax(dim=1).cpu())

                    val_loss_seg.append(seg_GHM_loss_fn(seg,target).item())
                    val_loss_edge.append(edge_GHM_loss_fn(edge,edge_target)+EMD_loss_fn(is_edge_prob,edge_target).item())
            
            # ema.restore()
            y_true=torch.cat(y_true,dim=-1)
            y_pred=torch.cat(y_pred,dim=-1)
            confusion_matrix=utils.confusion_matrix(vocab['<vocab_size>'],y_pred,y_true)
            val_l1=utils.cal_macro_F1(confusion_matrix)
            val_l1_total=torch.mean(torch.tensor(val_l1))
            writer.add_scalar('Loss/valid/L1_score', val_l1_total, i)

            recall_matrix=np.zeros_like(confusion_matrix)
            for j in range(vocab['<vocab_size>']):
                recall_matrix[j,:]=confusion_matrix[j,:]/(confusion_matrix[j,:].sum()+1e-10)
            writer.add_figure('recall', utils.plot_confusion_matrix(recall_matrix), i)
            
            precision_matrix=np.zeros_like(confusion_matrix)
            for j in range(vocab['<vocab_size>']):
                precision_matrix[:,j]=confusion_matrix[:,j]/(confusion_matrix[:,j].sum()+1e-10)
            writer.add_figure('precision', utils.plot_confusion_matrix(precision_matrix), i)

            val_loss_seg_total=torch.mean(torch.tensor(val_loss_seg))
            writer.add_scalar('Loss/valid/seg', val_loss_seg_total, i)
            val_loss_edge_total=torch.mean(torch.tensor(val_loss_edge))
            writer.add_scalar('Loss/valid/edge', val_loss_edge_total, i)
        
        if i%config.test_interval==0:
            # pass
            print('testing...')
            model.eval()
            id=1

            for path, subdirs, files in os.walk(os.path.join('data','test')):
                for file in files:
                    if file=='transcriptions.csv':
                        trans=pd.read_csv(os.path.join(path,file))
                        trans['path'] = trans.apply(lambda x: os.path.join(path,'wavs', x['name']+'.wav'), axis=1)
                        
                        ph_confidence_total=[]
                        for idx in trange(len(trans)):
                            ph_seq_pred,ph_dur_pred,ph_confidence,plot1,plot2=infer_once(
                                trans.loc[idx,'path'],
                                trans.loc[idx,'ph_seq'].split(' '),
                                model,
                                return_plot=True)
                            
                            writer.add_figure(f'{id}/melseg', plot1, i)
                            writer.add_figure(f'{id}/probvec', plot2, i)
                            id+=1

                            ph_confidence_total.append(ph_confidence)
                        writer.add_scalar('Accuracy/test_confidence', np.mean(ph_confidence_total), i)
                        
        
        if i%config.save_ckpt_interval==0 and i != 0:
            # ema.apply_shadow()
            torch.save(model.state_dict(), f'ckpt/{model_name}_{i}.pth')
            # ema.restore()
            print(f'saved model at {i} steps, path: ckpt/{model_name}_{i}.pth')

    progress_bar.close()