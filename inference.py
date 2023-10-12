from typing import Any
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
import librosa
from abc import ABCMeta, abstractmethod
import pandas as pd

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    vocab = yaml.safe_load(file)


class Dictionary(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self)->None:
        pass

    @abstractmethod
    def load(self, file_path:str)->None:
        pass

    @abstractmethod
    def __call__(self, word:str)->list:
        pass

    @abstractmethod
    def __contains__(self, item)->bool:
        pass

class DataItem(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        self.name:str
        self.path:str
        self.input_audio:torch.Tensor
        self.input_seq:np.ndarray
        self.aligned_tg=None

    def align(self,aligner,*args, **kwargs):
        # aligner:Aligner
        aligned_res,align_information=aligner(self.input_audio,self.input_seq,*args, **kwargs)
        if self.aligned_tg is not None:
            self.aligned_tg=aligned_res
        else:
            self.aligned_tg['phones']=aligned_res['phones']
        self.postprocess_align()
        return align_information

    @abstractmethod
    def postprocess_align(self):
        pass

    def postprocess_tg(self,tg_processor,*args,**kwargs):
        # tg_processor:TextGridProcessor
        self.aligned_tg=tg_processor(self,*args,**kwargs)

    def get_align_result(self)->tg.TextGrid:
        if self.aligned_tg is None:
            raise Exception('You should align first!')
        else:
            return self.aligned_tg

    def save_align_result(self,folder:str):
        if self.aligned_tg is None:
            raise Exception('You should align first!')
        else:
            if not os.path.exists(folder):
                os.mkdir(folder)
            self.aligned_tg.write(os.path.join(folder,self.name+'.TextGrid'))

class TextGridProcessor(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self,data_item:DataItem, *args: Any, **kwargs: Any) -> Any:
        pass

class AddAspiration(TextGridProcessor):
    def __init__(self) -> None:
        super().__init__()
    
    def get_ap_from_audio(self,audio:torch.Tensor):
        ap_interval=get_ap_interval(audio)
        ap_tier=intervals_to_tier(ap_interval,['AP'])
        return ap_tier
    def tier_intersection(self,tier1,tier2,text='AP'):
        intersection_tier=[]
        for i in tier1:
            for j in tier2:
                if i.xmax>j.xmin and i.xmin<j.xmax:
                    intersection_tier.append(tg.Interval(text=text,xmin=max(i.xmin,j.xmin),xmax=min(i.xmax,j.xmax)))
        return tg.Tier(intersection_tier)
    
    def add_intervals(self,tier,interval):
        tier.extend(interval)
        tier=sorted(tier,key=lambda x:x.xmin)
        return tier

    def __call__(self,data_item:DataItem, *args: Any, **kwargs: Any) -> Any:
        tier_of_ap=self.get_ap_from_audio(data_item.input_audio)
        tier_of_empty=tg.Tier([i for i in data_item.aligned_tg['phones'] if i.text==''])
        intersection_tier=self.tier_intersection(tier_of_ap,tier_of_empty)
        intersection_tier=tg.Tier([i for i in intersection_tier if i.xmax-i.xmin>config.postprocess.min_interval_dur])
        data_item.aligned_tg['words']=[i for i in data_item.aligned_tg['words'] if i.text !='']
        data_item.aligned_tg['phones']=[i for i in data_item.aligned_tg['phones'] if i.text !='']
        data_item.aligned_tg['words']=self.add_intervals(data_item.aligned_tg['words'],intersection_tier)
        data_item.aligned_tg['phones']=self.add_intervals(data_item.aligned_tg['phones'],intersection_tier)
        data_item.aligned_tg['words']=fill_in_empty_of_tier(data_item.aligned_tg['words'],length=data_item.input_audio.shape[-1]/config.sample_rate,text='SP')
        data_item.aligned_tg['phones']=fill_in_empty_of_tier(data_item.aligned_tg['phones'],length=data_item.input_audio.shape[-1]/config.sample_rate,text='SP')

        return data_item.aligned_tg


class WordDictionary(Dictionary):
    def __init__(self):
        self.dict={}
    
    def load(self, file_path:str):
        assert os.path.exists(file_path),f'{file_path} not found!'
        with open(file_path,'r') as f:
            dictionary=f.readlines()
        self.dict={i.split('\t')[0].strip():i.split('\t')[1].strip().split(' ') for i in dictionary}
    
    def __call__(self, word:str):
        if word in self.dict:
            return self.dict[word]
        else:
            raise Exception(f'{word} not in dictionary!')

    def __contains__(self, item):
        return item in self.dict


class PhonemeDictionary(Dictionary):
    def __init__(self):
        pass
    
    def load(self, file_path:str):
        Warning('PhonemeDictionary do not need to load file!')
    
    def __call__(self, word:str):
        return [word]
    
    def __contains__(self, item):
        return True

    def g2p(self, words: list) -> list:
        return words


class Aligner:
    def __init__(self,model) -> None:
        self.model=model

    def decode_ctc(self,ctc_pred):
        ctc_pred=ctc_pred.cpu().numpy()
        ctc_pred=np.argmax(ctc_pred,axis=0)
        ctc_seq=[]
        for idx in range(len(ctc_pred)-1):
            if ctc_pred[idx]!=ctc_pred[idx+1]:
                ctc_seq.append(ctc_pred[idx])
        ctc_ph_seq=[vocab[i] for i in ctc_seq if i != 0]

        return ctc_ph_seq

    def decode_alignment(self,ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log):
        # 乘上is_phoneme正确分类的概率
        prob_log[0,:]+=prob_log[0,:]
        prob_log[1:,:]+=1/prob_log[[0],:]

        # forward
        dp=np.zeros([len(ph_seq_num),prob_log.shape[-1]])-np.inf
        #只能从<EMPTY>开始或者从第一个音素开始
        dp[0,0]=prob_log[ph_seq_num[0],0]
        if ph_seq_num[0]==0:
            dp[1,0]=prob_log[ph_seq_num[1],0]
        backtrack_j=np.zeros_like(dp)-1
        for i in range(1,dp.shape[-1]):
            # [j,i-1]->[j,i]
            prob1=dp[:,i-1]+prob_log[ph_seq_num[:],i]+not_edge_prob_log[i]-config.infer.empty_number_punish*(ph_seq_num[:]==0)
            # [j-1,i-1]->[j,i]
            prob2=dp[:-1,i-1]+prob_log[ph_seq_num[1:],i]+is_edge_prob_log[i]
            prob2+=-config.infer.empty_number_punish*(ph_seq_num[1:]==0)
            prob2=np.concatenate([np.array([-np.inf]),prob2])
            # [j-2,i-1]->[j,i]
            # 不能跳过音素，可以跳过<EMPTY>
            prob3=dp[:-2,i-1]+prob_log[ph_seq_num[2:],i]+is_edge_prob_log[i]
            prob3[ph_seq_num[1:-1]!=0]=-np.inf
            prob3=np.concatenate([np.array([-np.inf,-np.inf]),prob3])

            backtrack_j[:,i]=np.arange(len(prob1))-np.argmax(np.stack([prob1,prob2,prob3],axis=0),axis=0)
            dp[:,i]=np.max(np.stack([prob1,prob2,prob3],axis=0),axis=0)
        backtrack_j=backtrack_j.astype(np.int32)

        # backward
        ph_seq_num_pred=[]
        ph_time_int=[]
        frame_confidence=[]
        #只能从最后一个音素或者<EMPTY>结束
        if ph_seq_num[-1]==0 and dp[-1,-1]<dp[-2,-1]:
            # confidence=dp[-2,-1]
            j=int(len(ph_seq_num)-2)
        else:
            # confidence=dp[-1,-1]
            j=int(len(ph_seq_num)-1)
        i=int(dp.shape[-1]-1)
        while j>=0:
            frame_confidence.append(dp[j,i])
            if j!=backtrack_j[j][i]:
                ph_seq_num_pred.append(int(ph_seq_num[j]))
                ph_time_int.append(i)
            j=backtrack_j[j][i]
            i-=1
        ph_seq_num_pred.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence=np.exp(np.diff(frame_confidence,1))

        return np.array(ph_seq_num_pred),np.array(ph_time_int),np.array(frame_confidence)

    def __call__(self,audio,ph_seq,return_confidence=False,return_ctc_pred=False,return_plot=False):

        melspec=utils.extract_normed_mel(audio)
        T=melspec.shape[-1]
        melspec=utils.pad_to_divisible_length(melspec,32)

        with torch.no_grad():
            h,seg,ctc,edge=self.model(melspec.to(config.device))
            seg,ctc,edge=seg[0,:,:T],ctc[0,:,:T],edge[0,:,:T]

        # postprocess output
        seg_prob=torch.nn.functional.softmax(seg,dim=0)
        # seg_prob[0,:]*=config.inference_empty_coefficient
        # seg_prob[0,:]*=torch.tensor(1-get_vowel_frame(audio.squeeze(0).cpu().numpy())).to(config.device)
        seg_prob/=seg_prob.sum(dim=0)

        prob_log=seg_prob.log().cpu().numpy()

        seg_prob=seg_prob.cpu().numpy()

        edge=torch.nn.functional.softmax(edge,dim=0)
        edge_pred=edge[0,:].clone().cpu().numpy()

        edge_diff=np.concatenate(([0],edge_pred,[1]))
        edge_diff=np.diff(edge_diff,1)
        edge_diff=edge_diff/2

        is_edge_prob=edge_pred
        is_edge_prob[1:]+=is_edge_prob[:-1]
        is_edge_prob=(is_edge_prob/2)**config.inference_edge_weight

        is_edge_prob_log=np.log(is_edge_prob.clip(1e-10,1-1e-10))

        not_edge_prob=1-is_edge_prob
        not_edge_prob_log=np.log(not_edge_prob)

        ph_seq_num=np.array([vocab[i] for i in ph_seq])

        # dynamic programming decoding
        ph_seq_num_pred,ph_time_pred_int,frame_confidence=self.decode_alignment(ph_seq_num,prob_log,is_edge_prob_log,not_edge_prob_log)

        # calculat time
        ph_time_pred=ph_time_pred_int.astype('float64')+(edge_diff[ph_time_pred_int]*config.inference_edge_weight).clip(-0.5,0.5)
        ph_time_pred=np.concatenate((ph_time_pred,[T+1]))*(config.hop_length/config.sample_rate)
        ph_time_pred[0]=0

        ph_dur_pred=np.diff(ph_time_pred,1)

        # ctc seq decode
        if return_ctc_pred:
            ctc_pred=torch.nn.functional.softmax(ctc,dim=0)
            ctc_ph_seq=self.decode_ctc(ctc_pred)

        ph_seq_pred=np.array([vocab[i] for i in ph_seq_num_pred])
        ph_interval_pred=np.stack([ph_time_pred[:-1],ph_time_pred[1:]],axis=1)
        phones_tier=intervals_to_tier(ph_interval_pred,ph_seq_pred,ignore_text_list=[vocab[0]])
        phones_tier=fill_in_empty_of_tier(phones_tier,T)
        
        aligned_tg=tg.TextGrid()
        aligned_tg['phones']=phones_tier
        align_information={}
        if return_confidence:
            align_information['confidence']=frame_confidence.mean()
        if return_ctc_pred:
            align_information['ctc_ph_seq']=ctc_ph_seq
        if return_plot:
            plot1=utils.plot_spectrogram_and_phonemes(melspec[0],target_gt=frame_confidence*config.n_mels,ph_seq=ph_seq_pred,ph_dur=ph_dur_pred)
            plot2=utils.plot_spectrogram_and_phonemes(seg_prob,target_gt=edge_pred*vocab['<vocab_size>'])#target_pred=frame_target,
            align_information['plot']=[plot1,plot2]
        return aligned_tg,align_information

class DataItemOfWordLab(DataItem):
    def __init__(self,file_path,dictionary:Dictionary) -> None:
        super().__init__()
        self.name=os.path.basename(file_path)
        self.path=file_path
        self.input_audio=utils.load_resampled_audio(file_path+'.wav')
        self.word_seq=self.read_lab(file_path+'.lab')
        self.input_seq,self.ph_num=self.word_to_ph(self.word_seq,dictionary)
        self.aligned_tg=tg.TextGrid({})
        self.aligned_tg['words']=tg.Tier([])
        self.aligned_tg['phones']=tg.Tier([])

    def read_lab(self,lab_path:str)->list:
        with open(lab_path,'r') as f:
            lab_seq=f.readlines()[0].strip().split(' ')
        return np.array(lab_seq)
    
    def word_to_ph(self,word_seq,dictionary:Dictionary):
        ph_seq=[vocab[0]]
        ph_num=[]
        for word in word_seq:
            ph_seq.extend(dictionary(word))
            ph_num.append(len(dictionary(word)))
            ph_seq.append(vocab[0])
        return np.array(ph_seq),np.array(ph_num)

    def add_words_tier(self,phones_tg,word_seq,ph_num):
        phones_tier=[i for i in phones_tg['phones'] if i.text!='']
        words_tier=[]
        ph_location=np.cumsum([0,*ph_num])
        for i in range(len(ph_location)-1):
            if i>0 and words_tier[-1].xmax!=phones_tier[ph_location[i]].xmin:
                words_tier.append(tg.Interval('',words_tier[-1].xmax,phones_tier[ph_location[i]].xmin))
            words_tier.append(tg.Interval(word_seq[i],phones_tier[ph_location[i]].xmin,phones_tier[ph_location[i+1]-1].xmax))
        # print(phones_tg)
        phones_tg['words']=tg.Tier(words_tier)
        return phones_tg
    
    def postprocess_align(self):
        self.aligned_tg=self.add_words_tier(self.aligned_tg,self.word_seq,self.ph_num)
        self.aligned_tg.move_to_end('phones')

def get_ap_interval(audio):
    audio=audio.cpu().numpy()
    if len(audio.shape)>1:
        audio=audio[0]

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=config.sample_rate, n_fft=2048, hop_length=config.hop_length).squeeze(0)
    not_LFNoise=spectral_centroid>config.postprocess.br_centroid

    rms_db=20*np.log10(librosa.feature.rms(y=audio,hop_length=config.hop_length)[0]/2e-5)
    not_space=rms_db>config.postprocess.br_db

    chromagram = librosa.feature.chroma_stft(y=audio, sr=config.sample_rate, hop_length=config.hop_length)
    chromagram=chromagram/np.sum(chromagram,axis=0)
    chromagram_entropy=-np.sum(chromagram*np.log(chromagram),axis=0)
    not_vowel=chromagram_entropy>config.postprocess.chromagram_entropy_thresh


    is_ap=1*not_LFNoise*not_vowel*not_space
    is_ap_diff=np.diff(is_ap,1)
    ap_interval=[]
    left_idx=-1
    for idx,is_interval_edge in enumerate(is_ap_diff):
        if is_interval_edge==1:
            left_idx=idx
        elif is_interval_edge==-1:
            if left_idx>=0 and (idx-left_idx)*(config.hop_length/config.sample_rate)>config.postprocess.min_interval_dur:
                ap_interval.append([left_idx,idx])
    ap_interval=np.array(ap_interval)

    return ap_interval*config.hop_length/config.sample_rate

def interval_intersection(intervals_a=[[1,3],[6,8]],intervals_b=[[2,6]]):
    idx_a=0
    idx_b=0
    intersection=[]
    interval_idx=[]
    while idx_a<len(intervals_a) and idx_b<len(intervals_b):
        if intervals_a[idx_a][0]>intervals_b[idx_b][1]:
            idx_b+=1
        elif intervals_a[idx_a][1]<intervals_b[idx_b][0]:
            idx_a+=1
        else:
            intersection.append([max(intervals_a[idx_a][0],intervals_b[idx_b][0]),min(intervals_a[idx_a][1],intervals_b[idx_b][1])])
            interval_idx.append([idx_a,idx_b])
            if intervals_a[idx_a][1]<intervals_b[idx_b][1]:
                idx_a+=1
            else:
                idx_b+=1
    
    return np.array(intersection),np.array(interval_idx)

def get_vowel_frame(audio):
    y=audio
    rms_db=20*np.log10(librosa.feature.rms(y=y,hop_length=config.hop_length)[0]/2e-5)
    not_space=rms_db>config.postprocess.vowel_db

    chromagram = librosa.feature.chroma_stft(y=y, sr=config.sample_rate, hop_length=config.hop_length)
    chromagram=chromagram/np.sum(chromagram,axis=0)
    chromagram_entropy=-np.sum(chromagram*np.log(chromagram),axis=0)
    not_ap=chromagram_entropy<config.postprocess.chromagram_entropy_thresh

    is_vowel=1*not_ap*not_space
    is_vowel_diff=np.diff(is_vowel,1)
    left_idx=-1
    for idx,is_interval_edge in enumerate(is_vowel_diff):
        if is_interval_edge==1:
            left_idx=idx
        elif is_interval_edge==-1:
            if (idx-left_idx)*(config.hop_length/config.sample_rate)<config.postprocess.min_interval_dur:
                is_vowel[left_idx+1:idx+1]=0
                left_idx=-1

    return is_vowel

def detect_AP(audio_path,ph_seq,ph_interval):
    # empty_pos=np.argwhere(ph_seq==vocab[0]).T[0]
    empty_interval=ph_interval[ph_seq==vocab[0],:]
    output_interval,interval_idx=interval_intersection(empty_interval,get_ap_interval(audio_path))
    if len(interval_idx)<1:
        return []
    
    output_interval=np.array([i for i in output_interval if i[1]-i[0]>config.postprocess.min_interval_dur])

    return output_interval


def load_model(model_path='ckpt'):
    # input: str:model_path
    # output: FullModel:model
    model=FullModel().to(config.device).eval()
    model_path=model_path
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

    return model

def intervals_to_tier(intervals,text,ignore_text_list=[]):
    if len(text)==1:
        text=[*text]*len(intervals)
    intervals_list=[]
    for i in range(len(text)):
        if text[i] in ignore_text_list:
            continue
        else:
            intervals_list.append(tg.Interval(text[i],intervals[i,0],intervals[i,1]))

    return tg.Tier(intervals_list)

def fill_in_empty_of_tier(tier,length,text=''):
    res=[]
    if tier[0].xmin>0:
        res.append(tg.Interval(text,0,tier[0].xmin))

    for idx,item in enumerate(tier):
        if item.text==text:
            res.append(tg.Interval(text,item.xmin,item.xmax))
        else:
            res.append(item)
        if idx<len(tier)-1 and item.xmax<tier[idx+1].xmin:
            res.append(tg.Interval(text,item.xmax,tier[idx+1].xmin))
    if tier[-1].xmax<length:
        res.append(tg.Interval(text,tier[-1].xmax,length))
    return tg.Tier(res)

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
    # input: config.yaml, wavs, labs, dictionary, phoneme_mode
    # ouput: textgrids
    args=parse_args()

    model=load_model(args.model_folder_path)

    if args.phoneme_mode:
        dictionary=PhonemeDictionary()
    else:
        dictionary=WordDictionary()
        dictionary.load(args.dictionary_path)
    
    aligner=Aligner(model)
    ap_adder=AddAspiration()

    for path, subdirs, files in os.walk(args.segments_path):
        if len(subdirs)==0:
            print(f'processing {path}...')

            file_names=list(
                set(map(lambda x:x[:-4],filter(lambda x:x.endswith('.lab'),files))) & 
                set(map(lambda x:x[:-4],filter(lambda x:x.endswith('.wav'),files)))
                )

            for file_name in tqdm(file_names):

                item=DataItemOfWordLab(os.path.join(path,file_name),dictionary)
                item.align(aligner)
                item.postprocess_tg(ap_adder)

                item.aligned_tg.write(os.path.join(path,file_name+'.TextGrid'))