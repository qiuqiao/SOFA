import torch
import torchaudio
import numpy as np
import utils
from utils import dict_to_namespace
import os
import torch
from model import FullModel
import yaml
import pandas as pd
from tqdm import trange
from itertools import chain

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    vocab = yaml.safe_load(file)

def alignment_decode(ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log):
    prob_log=np.array(prob_log.cpu())
    is_edge_prob_log=np.array(is_edge_prob_log.cpu())
    not_edge_prob_log=np.array(not_edge_prob_log.cpu())

    dp=np.zeros([len(ph_seq_num),prob_log.shape[-1]])-np.inf
    #只能从<EMPTY>开始或者从第一个音素开始
    dp[0,0]=prob_log[ph_seq_num[0],0]
    dp[1,0]=prob_log[ph_seq_num[1],0]
    backtrack_j=np.zeros_like(dp)-1
    for i in range(1,dp.shape[-1]):
        # [j,i-1]->[j,i]
        prob1=dp[:,i-1]+prob_log[ph_seq_num[:],i]+not_edge_prob_log[i]
        # [j-1,i-1]->[j,i]
        prob2=dp[:-1,i-1]+prob_log[ph_seq_num[1:],i]+is_edge_prob_log[i]-config.inference_empty_punish
        prob2=np.concatenate([np.array([-np.inf]),prob2])
        # [j-2,i-1]->[j,i]
        prob3=dp[:-2,i-1]+prob_log[ph_seq_num[2:],i]+is_edge_prob_log[i]
        prob3=np.concatenate([np.array([-np.inf,-np.inf]),prob3])
        prob3[::2]=-np.inf# 不能跳过音素，可以跳过<EMPTY>

        backtrack_j[:,i]=np.arange(len(prob1))-np.argmax(np.stack([prob1,prob2,prob3],axis=0),axis=0)
        dp[:,i]=np.max(np.stack([prob1,prob2,prob3],axis=0),axis=0)

    backtrack_j=backtrack_j.astype(np.int32)
    target=[]
    #只能从最后一个音素或者<EMPTY>结束
    if dp[-1,-1]>=dp[-2,-1]:
        j=int(len(ph_seq_num)-1)
    else:
        j=int(len(ph_seq_num)-2)
    i=int(dp.shape[-1]-1)
    while j>=0:
        target.append(int(ph_seq_num[j]))
        j=backtrack_j[j][i]
        i-=1
        # print(i,j)
    target.reverse()
    
    return dp,backtrack_j,target

def infer_once(audio_path,ph_seq,model,return_plot=False):
    # extract melspec
    audio, sample_rate = torchaudio.load(audio_path)
    audio_peak=torch.max(torch.abs(audio))
    audio=audio/audio_peak*0.08
    if sample_rate!=config.sample_rate:
        audio=torchaudio.transforms.Resample(sample_rate, config.sample_rate)(audio)
    melspec=utils.extract_normed_mel(audio)
    padding_len=32-melspec.shape[-1]%32
    if padding_len==0:
        padding_len=32
    melspec=torch.nn.functional.pad(melspec,(0,padding_len))

    # forward
    with torch.no_grad():
        h,seg,ctc,edge=model(melspec.to(config.device))

        seg_prob=torch.nn.functional.softmax(seg[0],dim=0)

        edge=torch.nn.functional.softmax(edge,dim=1)

        edge_pred=edge[0,0,:].clone()

        edge_diff=edge_pred.clone()
        edge_diff[1:]-=edge_diff[:-1].clone()
        edge_diff=edge_diff/2

        is_edge_prob=edge_pred
        is_edge_prob[1:]+=is_edge_prob[:-1].clone()
        is_edge_prob=(is_edge_prob/2)**config.inference_edge_weight

        is_edge_prob_log=torch.log(is_edge_prob.clamp(1e-10,1-1e-10))

        not_edge_prob=1-is_edge_prob
        not_edge_prob_log=torch.log(not_edge_prob)
    
    ph_seq_num=[vocab[i] for i in ph_seq if vocab[i] != 0]
    zeros=[0]*(len(ph_seq_num))
    ph_seq_num=list(chain.from_iterable(zip(zeros,ph_seq_num)))
    ph_seq_num.append(0)

    # dynamic programming decoding
    prob_log=seg_prob.log()
    dp,backtrack_j,target=alignment_decode(ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log)

    ph_seq_pred=[]
    ph_time_pred_int=[]
    ph_time_pred=[]
    ph_dur_pred=[]
    for idx, ph_num in enumerate(target):
        if idx==0:
            ph_seq_pred.append(vocab[int(ph_num)])
            ph_time_pred.append(0)
        else:
            if ph_num!=target[idx-1]:
                ph_seq_pred.append(vocab[int(ph_num)])
                ph_time_pred_int.append(idx)
                ph_time_pred.append(idx+(edge_diff[idx]*config.inference_edge_weight).clamp(-0.5,0.5))
                ph_dur_pred.append(ph_time_pred[-1]-ph_time_pred[-2])
    ph_dur_pred.append(len(target)-ph_time_pred[-1])
    ph_dur_pred=(torch.tensor(ph_dur_pred))*torch.tensor(config.hop_length/config.sample_rate)

    # ctc decoding
    ctc_pred=torch.nn.functional.softmax(ctc[0],dim=0)
    ctc_pred=ctc_pred.cpu().numpy()
    ctc_pred=np.argmax(ctc_pred,axis=0)
    ctc_seq=[]
    for idx in range(len(ctc_pred)-1):
        if ctc_pred[idx]!=ctc_pred[idx+1]:
            ctc_seq.append(ctc_pred[idx])
    ctc_ph_seq=[vocab[i] for i in ctc_seq if i != 0]

    # calculating confidence
    ph_time_pred=torch.cat((torch.tensor([0.]),torch.tensor(ph_time_pred),torch.tensor([seg_prob.shape[-1]]).float()),dim=0)
    ph_time_pred_int=torch.cat((torch.tensor([0]),torch.tensor(ph_time_pred_int),torch.tensor([seg_prob.shape[-1]])),dim=0).round().int()

    ph_confidence=[]
    frame_confidence=np.zeros([seg_prob.shape[-1]])
    for i in range(len(ph_seq_pred)):
        conf_seg=seg_prob[vocab[ph_seq_pred[i]]][ph_time_pred_int[i]:ph_time_pred_int[i+1]].cpu().numpy().mean()
        if ph_time_pred_int[i+1]-ph_time_pred_int[i]>2:
            conf_edge=0.5*(not_edge_prob[ph_time_pred_int[i]+1:ph_time_pred_int[i+1]-1].cpu().numpy().mean())
            conf_edge+=0.5*(is_edge_prob[ph_time_pred_int[i]].cpu().numpy()+is_edge_prob[ph_time_pred_int[i+1]-1].cpu().numpy())/2
        else:
            conf_edge=(is_edge_prob[ph_time_pred_int[i]].cpu().numpy()+is_edge_prob[ph_time_pred_int[i+1]-1].cpu().numpy())
        conf_curr=np.sqrt(conf_seg*conf_edge)
        # if ph_time_pred_int[i+1]-ph_time_pred_int[i]>2:
        #     conf_curr+=0.25*(not_edge_prob[ph_time_pred_int[i]+1:ph_time_pred_int[i+1]-1].cpu().numpy().mean())
        #     conf_curr+=0.25*(is_edge_prob[ph_time_pred_int[i]].cpu().numpy()+is_edge_prob[ph_time_pred_int[i+1]-1].cpu().numpy())
        # else:
        #     conf_curr+=0.5*(is_edge_prob[ph_time_pred_int[i]].cpu().numpy()+is_edge_prob[ph_time_pred_int[i+1]-1].cpu().numpy())

        if not conf_curr>0: #出现nan时改为0
            conf_curr=0
        frame_confidence[ph_time_pred_int[i]:ph_time_pred_int[i+1]]=conf_curr
        ph_confidence.append(conf_curr)

    if not return_plot:
        return ph_seq_pred,ph_dur_pred.numpy(),ctc_ph_seq,np.mean(ph_confidence)
    
    else:
        return ph_seq_pred,ph_dur_pred.numpy(),ctc_ph_seq,np.mean(ph_confidence),\
                utils.plot_spectrogram_and_phonemes(melspec[0],target_pred=frame_confidence*config.n_mels,ph_seq=ph_seq_pred,ph_dur=ph_dur_pred.numpy()),\
                utils.plot_spectrogram_and_phonemes(seg_prob.cpu(),target_pred=target,target_gt=edge_pred.cpu().numpy()*vocab['<vocab_size>'])

if __name__ == '__main__':
    # load model
    model=FullModel().to(config.device)
    model_name=config.model_name
    ckpt_list=os.listdir('ckpt')
    ckpt_list=[i for i in ckpt_list if i.startswith(model_name)]
    ckpt_list.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
    ckpt_name=ckpt_list[-1]
    print(f'loading {ckpt_name}...')
    model.load_state_dict(torch.load(f'ckpt/{ckpt_name}'))

    # inference all data
    for path, subdirs, files in os.walk(config.inference_data_path):
        for file in files:
            if file=='transcriptions.csv':
                trans=pd.read_csv(os.path.join(path,file))
                trans['path'] = trans.apply(lambda x: os.path.join(path,'wavs', x['name']+'.wav'), axis=1)

                ph_seqs=[]
                ph_durs=[]
                ctc_ph_seqs=[]
                ph_confidences=[]
                for idx in trange(len(trans)):
                    ph_seq_pred,ph_dur_pred,ctc_ph_seq,ph_confidence=infer_once(trans.loc[idx,'path'],trans.loc[idx,'ph_seq'].split(' '),model)
                    ph_seqs.append(' '.join(ph_seq_pred))
                    ph_durs.append(' '.join([str(i) for i in ph_dur_pred]))
                    ctc_ph_seqs.append(' '.join(ctc_ph_seq))
                    ph_confidences.append(ph_confidence)
                
                trans['ph_seq']=ph_seqs
                trans['ph_dur']=ph_durs
                trans['ctc_ph_seq']=ctc_ph_seqs
                trans['confidence']=ph_confidences

                trans.to_csv(os.path.join(path,'transcriptions.csv'),index=False)