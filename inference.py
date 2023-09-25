import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import utils
from utils import dict_to_namespace
import os
import torch
from model import FullModel
from data_augmentation import AudioAugmentation
import yaml
from argparse import Namespace
import pandas as pd
from tqdm import trange
from itertools import chain
import numba as nb

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    vocab = yaml.safe_load(file)

def alignment_decode(ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log,inference_edge_weight=config.inference_edge_weight):
    prob_log=np.array(prob_log.cpu())
    is_edge_prob_log=np.array(is_edge_prob_log.cpu())
    not_edge_prob_log=np.array(not_edge_prob_log.cpu())
    dp=np.zeros([len(ph_seq_num),prob_log.shape[-1]])-np.inf
    for j in range(len(ph_seq_num)):
        dp[j,0]=prob_log[ph_seq_num[j],0]
    backtrack_j=np.zeros_like(dp)-1
    for i in range(1,dp.shape[-1]):
        # [j,i-1]->[j,i]
        prob1=dp[:,i-1]+prob_log[ph_seq_num[:],i]+not_edge_prob_log[i]*inference_edge_weight
        # [j-1,i-1]->[j,i]
        prob2=dp[:-1,i-1]+prob_log[ph_seq_num[1:],i]+is_edge_prob_log[i]*inference_edge_weight
        prob2=np.concatenate([np.array([-np.inf]),prob2])
        # [j-2,i-1]->[j,i]
        prob3=dp[:-2,i-1]+prob_log[ph_seq_num[2:],i]+is_edge_prob_log[i]*inference_edge_weight
        prob3=np.concatenate([np.array([-np.inf,-np.inf]),prob3])
        prob3[::2]=-np.inf

        backtrack_j[:,i]=np.arange(len(prob1))-np.argmax(np.stack([prob1,prob2,prob3],axis=0),axis=0)
        dp[:,i]=np.max(np.stack([prob1,prob2,prob3],axis=0),axis=0)

    backtrack_j=backtrack_j.astype(np.int32)
    target=[]
    j=int(len(ph_seq_num)-1)
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

    # inference
    with torch.no_grad():
        h,seg,ctc,edge=model(melspec.to(config.device))
        seg_prob=torch.nn.functional.softmax(seg[0],dim=0)
        edge=torch.nn.functional.softmax(edge,dim=1)
        is_edge_prob=edge[0,1,:]
        is_edge_prob_log=torch.log(is_edge_prob)
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
    ph_time_pred=[]
    ph_dur_pred=[]
    for idx, ph_num in enumerate(target):
        if idx==0:
            ph_seq_pred.append(vocab[int(ph_num)])
            ph_time_pred.append(0)
        else:
            if ph_num!=target[idx-1]:
                ph_seq_pred.append(vocab[int(ph_num)])
                ph_time_pred.append(idx)
                ph_dur_pred.append(ph_time_pred[-1]-ph_time_pred[-2])
    ph_dur_pred.append(len(target)-ph_time_pred[-1])
    ph_dur_pred=((torch.tensor(ph_dur_pred))*config.hop_length/config.sample_rate)

    # calculating confidence
    ph_time_pred.append(seg_prob.shape[-1])
    frame_confidence=np.zeros([seg_prob.shape[-1]])
    ph_confidence=[]
    for i in range(len(ph_time_pred)-1):
        conf_curr=0.5*(seg_prob[vocab[ph_seq_pred[i]]][ph_time_pred[i]:ph_time_pred[i+1]].cpu().numpy().mean())
        conf_curr+=0.25*(not_edge_prob[ph_time_pred[i]+1:ph_time_pred[i+1]].cpu().numpy().mean())
        if ph_time_pred[i+1]>=is_edge_prob.shape[-1]:
            conf_curr+=0.25*(is_edge_prob[ph_time_pred[i]].cpu().numpy().mean())
        elif ph_time_pred[i]==0:
            # print(i,ph_time_pred,ph_time_pred[i],ph_seq_pred)
            conf_curr+=0.25*(is_edge_prob[ph_time_pred[i+1]].cpu().numpy().mean())
        else:
            conf_curr+=0.125*(is_edge_prob[ph_time_pred[i+1]].cpu().numpy().mean())
            conf_curr+=0.125*(is_edge_prob[ph_time_pred[i]].cpu().numpy().mean())

        frame_confidence[ph_time_pred[i]:ph_time_pred[i+1]]=conf_curr
        ph_confidence.append(conf_curr)
    
    if not return_plot:
        return ph_seq_pred,ph_dur_pred.numpy(),np.mean(ph_confidence)
    
    else:
        return ph_seq_pred,ph_dur_pred.numpy(),np.mean(ph_confidence),\
                utils.plot_spectrogram_and_phonemes(melspec[0],target_pred=frame_confidence*config.n_mels,ph_seq=ph_seq_pred,ph_dur=ph_dur_pred.numpy()),\
                utils.plot_spectrogram_and_phonemes(seg_prob.cpu(),target_pred=target,target_gt=is_edge_prob.cpu().numpy()*vocab['<vocab_size>'])

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
                ph_confidences=[]
                for idx in trange(len(trans)):
                    ph_seq_pred,ph_dur_pred,ph_confidence=infer_once(trans.loc[idx,'path'],trans.loc[idx,'ph_seq'].split(' '),model)
                    ph_seqs.append(' '.join(ph_seq_pred))
                    ph_durs.append(' '.join([str(i) for i in ph_dur_pred]))
                    ph_confidences.append(ph_confidence)
                
                trans['ph_seq']=ph_seqs
                trans['ph_dur']=ph_durs
                trans['confidence']=ph_confidences

                trans.to_csv(os.path.join(path,'transcriptions.csv'),index=False)