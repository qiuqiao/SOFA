import torch
import torchaudio
import numpy as np
import utils
from utils import dict_to_namespace
import os
import torch
from model import FullModel
import yaml
from tqdm import trange,tqdm
import argparse
import textgrids as tg

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    vocab = yaml.safe_load(file)

def alignment_decode(ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log):
    prob_log=np.array(prob_log)
    is_edge_prob_log=np.array(is_edge_prob_log)
    not_edge_prob_log=np.array(not_edge_prob_log)
    ph_seq_num=np.array(ph_seq_num)

    dp=np.zeros([len(ph_seq_num),prob_log.shape[-1]])-np.inf
    #只能从<EMPTY>开始或者从第一个音素开始
    dp[0,0]=prob_log[ph_seq_num[0],0]
    if ph_seq_num[0]==0:
        dp[1,0]=prob_log[ph_seq_num[1],0]
    backtrack_j=np.zeros_like(dp)-1
    for i in range(1,dp.shape[-1]):
        # [j,i-1]->[j,i]
        prob1=dp[:,i-1]+prob_log[ph_seq_num[:],i]+not_edge_prob_log[i]
        # [j-1,i-1]->[j,i]
        prob2=dp[:-1,i-1]+prob_log[ph_seq_num[1:],i]+is_edge_prob_log[i]
        prob2=np.concatenate([np.array([-np.inf]),prob2])
        prob2-=config.inference_empty_punish*(ph_seq_num==0)
        prob2[1:]-=config.inference_empty_punish*(ph_seq_num[:-1]==0)
        # [j-2,i-1]->[j,i]
        # 不能跳过音素，可以跳过<EMPTY>
        prob3=dp[:-2,i-1]+prob_log[ph_seq_num[2:],i]+is_edge_prob_log[i]
        prob3[ph_seq_num[1:-1]!=0]=-np.inf
        prob3=np.concatenate([np.array([-np.inf,-np.inf]),prob3])

        backtrack_j[:,i]=np.arange(len(prob1))-np.argmax(np.stack([prob1,prob2,prob3],axis=0),axis=0)
        dp[:,i]=np.max(np.stack([prob1,prob2,prob3],axis=0),axis=0)

    backtrack_j=backtrack_j.astype(np.int32)
    ph_seq_num_pred=[]
    ph_time_int=[]
    #只能从最后一个音素或者<EMPTY>结束
    if ph_seq_num[-1]==0 and dp[-1,-1]<dp[-2,-1]:
        j=int(len(ph_seq_num)-2)
    else:
        j=int(len(ph_seq_num)-1)
    i=int(dp.shape[-1]-1)
    while j>=0:
        if j!=backtrack_j[j][i]:
            ph_seq_num_pred.append(int(ph_seq_num[j]))
            ph_time_int.append(i)
        j=backtrack_j[j][i]
        i-=1
    ph_seq_num_pred.reverse()
    ph_time_int.reverse()
    
    return np.array(ph_seq_num_pred),np.array(ph_time_int)

def infer_once(audio_path,ph_seq,model,return_time=False,return_confidence=False,return_ctc_pred=False,return_plot=False):
    # extract melspec
    audio, sample_rate = torchaudio.load(audio_path)
    audio_peak=torch.max(torch.abs(audio))
    audio=audio/audio_peak*0.08
    if sample_rate!=config.sample_rate:
        audio=torchaudio.transforms.Resample(sample_rate, config.sample_rate)(audio)
    melspec=utils.extract_normed_mel(audio)
    T=melspec.shape[-1]
    padding_len=32-melspec.shape[-1]%32
    if padding_len==0:
        padding_len=32
    melspec=torch.nn.functional.pad(melspec,(0,padding_len))

    # forward
    with torch.no_grad():
        h,seg,ctc,edge=model(melspec.to(config.device))
        
        seg_prob=torch.nn.functional.softmax(seg[0],dim=0)
        prob_log=seg_prob.log().cpu().numpy()
        seg_prob=seg_prob.cpu().numpy()
        edge=torch.nn.functional.softmax(edge,dim=1)

        edge_pred=edge[0,0,:].clone().cpu().numpy()

        edge_diff=np.concatenate(([0],edge_pred,[1]))
        edge_diff=np.diff(edge_diff,1)
        edge_diff=edge_diff/2

        is_edge_prob=edge_pred
        is_edge_prob[1:]+=is_edge_prob[:-1]
        is_edge_prob=(is_edge_prob/2)**config.inference_edge_weight

        is_edge_prob_log=np.log(is_edge_prob.clip(1e-10,1-1e-10))

        not_edge_prob=1-is_edge_prob
        not_edge_prob_log=np.log(not_edge_prob)

    ph_seq_num=[vocab[i] for i in ph_seq]

    # dynamic programming decoding
    ph_seq_num_pred,ph_time_pred_int=alignment_decode(ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log)

    # calculating time
    ph_time_pred=ph_time_pred_int.astype('float64')
    ph_time_pred+=(edge_diff[ph_time_pred_int]*config.inference_edge_weight).clip(-0.5,0.5)
    ph_time_pred=np.concatenate((ph_time_pred,[T+1]))*(config.hop_length/config.sample_rate)
    ph_time_pred[0]=0

    ph_dur_pred=np.diff(ph_time_pred,1)

    # ctc decoding
    if return_ctc_pred:
        ctc_pred=torch.nn.functional.softmax(ctc[0],dim=0)
        ctc_pred=ctc_pred.cpu().numpy()
        ctc_pred=np.argmax(ctc_pred,axis=0)
        ctc_seq=[]
        for idx in range(len(ctc_pred)-1):
            if ctc_pred[idx]!=ctc_pred[idx+1]:
                ctc_seq.append(ctc_pred[idx])
        ctc_ph_seq=[vocab[i] for i in ctc_seq if i != 0]

    # calculating confidence
    if return_confidence or return_plot:

        ph_time_pred_int=torch.cat((torch.tensor([0]),torch.tensor(ph_time_pred_int),torch.tensor([seg_prob.shape[-1]])),dim=0).round().int()

        ph_confidence=[]
        frame_confidence=np.zeros([seg_prob.shape[-1]])
        frame_target=np.zeros([seg_prob.shape[-1]])
        for i in range(len(ph_seq_num_pred)):
            conf_seg=seg_prob[ph_seq_num_pred[i]][ph_time_pred_int[i]:ph_time_pred_int[i+1]].mean()
            if ph_time_pred_int[i+1]-ph_time_pred_int[i]>2:
                conf_edge=0.5*(not_edge_prob[ph_time_pred_int[i]+1:ph_time_pred_int[i+1]-1].mean())
                conf_edge+=0.5*(is_edge_prob[ph_time_pred_int[i]]+is_edge_prob[ph_time_pred_int[i+1]-1])/2
            else:
                conf_edge=(is_edge_prob[ph_time_pred_int[i]]+is_edge_prob[ph_time_pred_int[i+1]-1])
            conf_curr=np.sqrt(conf_seg*conf_edge)

            if not conf_curr>0: #出现nan时改为0
                conf_curr=0
            frame_confidence[ph_time_pred_int[i]:ph_time_pred_int[i+1]]=conf_curr
            frame_target[ph_time_pred_int[i]:ph_time_pred_int[i+1]]=ph_seq_num_pred[i]
            ph_confidence.append(conf_curr)

    ph_seq_pred=np.array([vocab[i] for i in ph_seq_num_pred])
    res=[ph_seq_pred,ph_dur_pred]
    if return_time:
        res.append(ph_time_pred)
    if return_confidence:
        res.append(np.mean(ph_confidence))
    if return_ctc_pred:
        res.append(ctc_ph_seq)
    if return_plot:
        plot1=utils.plot_spectrogram_and_phonemes(melspec[0],target_pred=frame_confidence*config.n_mels,ph_seq=ph_seq_pred,ph_dur=ph_dur_pred)
        plot2=utils.plot_spectrogram_and_phonemes(seg_prob,target_pred=frame_target,target_gt=edge_pred*vocab['<vocab_size>'])
        res.extend([plot1,plot2])
    return res
                                
def parse_args():
    """
    进行参数的解析
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-m','--model_folder_path', type=str, default='ckpt', help='model folder path. It should contain three files: a file with the .pth extension, config.yaml, and vocab.yaml.')
    parser.add_argument('-s','--segments_path', type=str, default='segments', help='segments path. It shoould contain 2 types of file: .wav and .lab')
    parser.add_argument('-d','--dictionary_path', type=str, default=os.path.join('dictionary','opencpop-extension.txt'), help='dictionary path. It should be a txt file.')
    parser.add_argument('-p','--phoneme_mode',action='store_true',help='phoneme mode. If this argument is set, the dictionary path will be ignored and the phoneme mode will be used.')

    return parser.parse_args()
    

if __name__ == '__main__':
    args=parse_args()

    # load model
    model=FullModel().to(config.device).eval()
    model_path=args.model_folder_path
    pth_list=os.listdir(model_path)
    pth_list=[i for i in pth_list if i.endswith('.pth')]
    if len(pth_list)==0:
        raise Exception('No .pth file in model folder!')
    elif len(pth_list)>1:
        raise Exception('More than one .pth file in model folder!')
    else:
        pth_name=pth_list[0]
    print(f'loading {pth_name}...')
    model.load_state_dict(torch.load(os.path.join(model_path,pth_name)))

    # if word mode, load dictionary
    if not args.phoneme_mode:
        assert os.path.exists(args.dictionary_path),f'{args.dictionary_path} not found!'
        with open(args.dictionary_path,'r') as f:
            dictionary=f.readlines()
        dictionary={i.split('\t')[0].strip():i.split('\t')[1].strip().split(' ') for i in dictionary}

    # inference all data
    for path, subdirs, files in os.walk(args.segments_path):
        if subdirs==[]:
            labs=[]
            wavs=[]
            for f in files:
                if f.endswith('.lab'):
                    labs.append(f)
                elif f.endswith('.wav'):
                    wavs.append(f)

            for lab in tqdm(labs):
                file_name=lab[:-4]
                if file_name+'.wav' not in wavs:
                    raise Exception(f'{os.path.join(path,file_name)}.wav not found!')
                else:
                    # if word mode, 根据词典生成音素序列
                    # 否则在lab文件中直接读取音素序列
                    # 同时在序列中加入<EMPTY>
                    with open(os.path.join(path,lab),'r') as f:
                        word_seq=f.readlines()[0].strip().split(' ')
                    ph_seq_input=[vocab[0]]
                    ph_num=[]
                    for word in word_seq:
                        if args.phoneme_mode:
                            ph_seq_input.append(word)
                        else:
                            if word in dictionary:
                                ph_seq_input.extend(dictionary[word])
                                ph_num.append(len(dictionary[word]))
                            else:
                                raise Exception(f'{word} not in dictionary!')
                        ph_seq_input.append(vocab[0])

                    ph_seq_pred,ph_dur_pred,ph_time_pred=infer_once(os.path.join(path,file_name+'.wav'),ph_seq_input,model,return_time=True)
                    ph_interval_pred=np.stack([ph_time_pred[:-1],ph_time_pred[1:]],axis=1)

                    # 去除<EMPTY>及其对应的ph_dur、ph_time
                    indexes_to_remove = np.where(ph_seq_pred==vocab[0])
                    ph_seq_pred = np.delete(ph_seq_pred, indexes_to_remove)
                    ph_interval_pred = np.delete(ph_interval_pred, indexes_to_remove,axis=0)

                    # convert to textgrid
                    textgrid=tg.TextGrid()
                    words=[]
                    phones=[]
                    ph_location=np.cumsum([0,*ph_num])
                    for i in range(len(ph_seq_pred)):
                        if i>0 and phones[-1].xmax!=ph_interval_pred[i,0]:
                            phones.append(tg.Interval('',phones[-1].xmax,ph_interval_pred[i,0])) 
                        phones.append(tg.Interval(ph_seq_pred[i],ph_interval_pred[i,0],ph_interval_pred[i,1]))
                    for i in range(len(ph_location)-1):
                        if i>0 and words[-1].xmax!=ph_interval_pred[ph_location[i],0]:
                            words.append(tg.Interval('',words[-1].xmax,ph_interval_pred[ph_location[i],0]))
                        words.append(tg.Interval(word_seq[i],ph_interval_pred[ph_location[i],0],ph_interval_pred[ph_location[i+1]-1,1]))
                    
                    textgrid['words']=tg.Tier(words)
                    textgrid['phones']=tg.Tier(phones)

                    textgrid.write(os.path.join(path,file_name+'.TextGrid'))

                    # with open(os.path.join(path,file_name+'.TextGrid'),'w') as f:
                    #     f.write(' '.join(ph_seq_pred)+'\n'+' '.join([str(i) for i in ph_dur_pred]))