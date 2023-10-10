import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import utils
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from utils import GaussianRampUpScheduler
from tqdm import tqdm, trange
from inference import infer_once
import random
import torch
import numpy as np
from dataset import FullLabelDataset, collate_fn, WeakLabelDataset, weak_label_collate_fn
from model import FullModel
from einops import rearrange
import yaml

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

    # dataset
    full_train_dataset = FullLabelDataset(name='train')
    full_train_dataloader = DataLoader(dataset=full_train_dataset, batch_size=config.batch_size_sup, shuffle=True, collate_fn=collate_fn)
    full_train_dataiter = iter(full_train_dataloader)

    full_valid_dataset = FullLabelDataset(name='valid')
    full_valid_dataloader = DataLoader(dataset=full_valid_dataset, batch_size=config.batch_size_sup, shuffle=False,collate_fn=collate_fn,drop_last=True)

    weak_train_dataset = WeakLabelDataset(name='train')
    weak_train_dataloader = DataLoader(dataset=weak_train_dataset, batch_size=config.batch_size_sup, shuffle=True, collate_fn=weak_label_collate_fn)
    weak_train_dataiter = iter(weak_train_dataloader)

    weak_valid_dataset = WeakLabelDataset(name='valid')
    weak_valid_dataloader = DataLoader(dataset=weak_valid_dataset, batch_size=config.batch_size_sup, shuffle=False,collate_fn=weak_label_collate_fn,drop_last=True)

    # usp_dataset = NoLabelDataset(name='train')
    # usp_dataloader = DataLoader(dataset=usp_dataset, batch_size=config.batch_size_usp, shuffle=True,collate_fn=collate_fn)
    # usp_dataiter = iter(usp_dataloader)

    # model
    # model=FullModel().to(config.device)
    model=FullModel().to(config.device)
    # model_path='ckpt'
    # pth_list=os.listdir(model_path)
    # pth_list=[i for i in pth_list if i.endswith('.pth')]
    # if len(pth_list)==0:
    #     raise Exception('No .pth file in model folder!')
    # elif len(pth_list)>1:
    #     raise Exception('More than one .pth file in model folder!')
    # else:
    #     pth_name=pth_list[0]
    # print(f'loading {pth_name}...')
    # model.load_state_dict(torch.load(os.path.join(model_path,pth_name)))

    # loss function
    seg_GHM_loss_fn=utils.GHMLoss(vocab['<vocab_size>'],num_prob_bins=10,alpha=0.999,label_smoothing=config.label_smoothing)
    edge_GHM_loss_fn=utils.GHMLoss(2,num_prob_bins=5,alpha=0.999999,label_smoothing=0.0,enable_prob_input=True)
    EMD_loss_fn=utils.BinaryEMDLoss()
    # CE_loss_fn=nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    # BCE_loss_fn=nn.BCELoss()
    MSE_loss_fn=nn.MSELoss()
    CTC_loss_fn = nn.CTCLoss()

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=config.max_steps)
    wsp_scheduler = GaussianRampUpScheduler(config.max_steps,0,config.max_steps)


    # start training
    model_name='model'
    progress_bar = tqdm(total=config.max_steps, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    writer = SummaryWriter()
    print('start training')
    for step in range(config.max_steps):
        model.train()
        optimizer.zero_grad()

        # full supervised
        # get data
        try:
            melspec,target,edge_target=next(full_train_dataiter)
        except StopIteration:
            full_train_dataiter = iter(full_train_dataloader)
            melspec,target,edge_target=next(full_train_dataiter)

        melspec=torch.tensor(melspec).to(config.device)
        target=torch.tensor(target).to(config.device).long().squeeze(1)
        edge_target=torch.tensor(edge_target).to(config.device).float()
        
        # forward
        h,seg,ctc,edge=model(melspec) 

        # calculate loss
        is_edge_prob=F.softmax(edge,dim=1)[:,0,:]
        seg_loss=seg_GHM_loss_fn(seg,target.squeeze(1))
        edge_diff_loss=MSE_loss_fn(torch.diff(is_edge_prob,dim=-1),torch.diff(edge_target[:,0,:],dim=-1))
        edge_GHM_loss=edge_GHM_loss_fn(edge,edge_target)
        edge_EMD_loss=0.01*EMD_loss_fn(is_edge_prob,edge_target[:,0,:])
        edge_loss=edge_GHM_loss+edge_EMD_loss+edge_diff_loss.mean()

        fsp_loss=edge_loss+seg_loss

        # log
        writer.add_scalar('Accuracy/train/fsp/accuracy', (seg.argmax(dim=1)==target).float().mean().item(), step)
        writer.add_scalar('Loss/train/fsp/seq', seg_loss.item(), step) 
        writer.add_scalar('Loss/train/fsp/edge', edge_loss.item(), step)


        # weak supervised
        # get data
        try:
            input_feature,ctc_target,ctc_target_lengths=next(weak_train_dataiter)
        except StopIteration:
            weak_train_dataiter = iter(weak_train_dataloader)
            input_feature,ctc_target,ctc_target_lengths=next(weak_train_dataiter)

        input_feature=torch.tensor(input_feature).to(config.device)
        ctc_target=torch.tensor(ctc_target).to(config.device).long()
        ctc_target_lengths=torch.tensor(ctc_target_lengths).to(config.device).long()

        # forward
        h,seg,ctc,edge=model(input_feature)

        # calculate loss
        ctc=F.log_softmax(ctc,dim=1)
        ctc=rearrange(ctc,'n c t -> t n c')
        wsp_loss=CTC_loss_fn(ctc, ctc_target, torch.tensor(ctc.shape[0]).repeat(ctc.shape[1]), ctc_target_lengths)

        # log
        writer.add_scalar('Loss/train/wsp/ctc', wsp_loss.item(), step) 

        # unsupervised training
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

        # writer.add_scalar('Loss/train/consistence', consistence_loss.item(), step)

        # loss+=usp_scheduler()*consistence_loss


        # sum up losses
        loss=fsp_loss+wsp_scheduler()*wsp_loss

        # backward and update
        loss.backward()
        optimizer.step()
        scheduler.step()
        wsp_scheduler.step()

        # log
        writer.add_scalar('Loss/train/total', loss.item(), step)
        writer.add_scalar('learning_rate/total', optimizer.param_groups[0]['lr'], step)
        writer.add_scalar('learning_rate/wsp', wsp_scheduler(), step)
        progress_bar.set_description(f'tr_loss: {loss.item():.3f}')
        progress_bar.update()

        if step%config.val_interval==0:
            # pass
            print('validating...')
            model.eval()

            # full supervised
            # forward
            y_true=[]
            y_pred=[]
            with torch.no_grad():
                for melspec,target,edge_target in full_valid_dataloader:
                    melspec,target,edge_target=melspec.to(config.device),target.to(config.device).squeeze(1),edge_target.to(config.device).squeeze(1).long()
                    h,seg,ctc,edge=model(melspec)

                    is_edge_prob=F.softmax(edge,dim=1)[:,0,:]

                    y_true.append(target.cpu())
                    y_pred.append(seg.argmax(dim=1).cpu())
            
            # log
            y_true=torch.cat(y_true,dim=-1)
            y_pred=torch.cat(y_pred,dim=-1)
            confusion_matrix=utils.confusion_matrix(vocab['<vocab_size>'],y_pred,y_true)
            val_F1=utils.cal_macro_F1(confusion_matrix)
            val_F1_total=torch.mean(torch.tensor(val_F1))
            writer.add_scalar('Accuracy/valid/F1_score', val_F1_total, step)

            recall_matrix=np.zeros_like(confusion_matrix)
            for j in range(vocab['<vocab_size>']):
                recall_matrix[j,:]=confusion_matrix[j,:]/(confusion_matrix[j,:].sum()+1e-10)
            writer.add_figure('recall', utils.plot_confusion_matrix(recall_matrix), step)
            
            precision_matrix=np.zeros_like(confusion_matrix)
            for j in range(vocab['<vocab_size>']):
                precision_matrix[:,j]=confusion_matrix[:,j]/(confusion_matrix[:,j].sum()+1e-10)
            writer.add_figure('precision', utils.plot_confusion_matrix(precision_matrix), step)


            # weak supervised
            # forward
            ctc_losses=[]
            with torch.no_grad():
                for input_feature,ctc_target,ctc_target_lengths in weak_valid_dataloader:
                    input_feature=input_feature.to(config.device)
                    ctc_target=ctc_target.to(config.device).long()
                    ctc_target_lengths=ctc_target_lengths.to(config.device).long()

                    h,seg,ctc,edge=model(input_feature)
                    ctc=F.log_softmax(ctc,dim=1)
                    ctc=rearrange(ctc,'n c t -> t n c')
                    ctc_losses.append(0.01*CTC_loss_fn(ctc, ctc_target, torch.tensor(ctc.shape[0]).repeat(ctc.shape[1]), ctc_target_lengths).cpu().item())
            ctc_loss_total=np.array(ctc_losses).mean()
            writer.add_scalar('Loss/valid/ctc', ctc_loss_total, step)
        
        if step%config.test_interval==0:
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
                            ph_seq_input=trans.loc[idx,'ph_seq'].split(' ')
                            new_lst = [vocab[0]] * (len(ph_seq_input) * 2 +1)  # 创建一个新列表，长度是原列表的2倍减1。
                            new_lst[1::2] = ph_seq_input  # 将原列表的元素插入到新列表的偶数位置。
                            ph_seq_input=new_lst
                            ph_seq_pred,ph_dur_pred,ph_confidence,ctc_ph_seq,plot1,plot2=infer_once(
                                trans.loc[idx,'path'],
                                ph_seq_input,
                                model,
                                return_confidence=True,
                                return_ctc_pred=True,
                                return_plot=True)
                            
                            writer.add_figure(f'{id}/melseg', plot1, step)
                            writer.add_figure(f'{id}/probvec', plot2, step)
                            writer.add_text(f'{id}/ctc_ph_seq', ' '.join(ctc_ph_seq), step)
                            id+=1

                            ph_confidence_total.append(ph_confidence)
                        writer.add_scalar('Accuracy/test/confidence', np.mean(ph_confidence_total), step)
                        
        
        if step%config.save_ckpt_interval==0 and step >= config.save_ckpt_start:
            torch.save(model.state_dict(), f'ckpt/{model_name}_{step}.pth')
            print(f'saved model at {step} steps, path: ckpt/{model_name}_{step}.pth')

    progress_bar.close()