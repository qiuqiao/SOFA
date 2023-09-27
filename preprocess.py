import torch
import torchaudio
import pandas as pd
import numpy as np
import utils
from utils import load_dict,save_dict,read_ndarray_from_bin,wirte_ndarray_to_bin
import os
import pickle
import yaml
from argparse import Namespace
from tqdm import tqdm,trange
import os
from einops import rearrange, reduce, repeat
from WavLM.WavLM import WavLM, WavLMConfig
from data_augmentation import AudioAugmentation

def dict_to_namespace(d):
    namespace = Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    vocab = yaml.safe_load(file)

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

def get_data_list(folder):
    data_list=pd.DataFrame()
    dataset_list=os.listdir(os.path.join('data', folder))
    for dataset_name in dataset_list:
        if os.path.exists(os.path.join('data', folder,dataset_name,'raw','transcriptions.csv')):
            trans=pd.read_csv(os.path.join('data', folder,dataset_name,'raw','transcriptions.csv'),dtype=str)
            trans['path'] = trans.apply(lambda name: os.path.join('data', folder,dataset_name,'raw','wavs',name['name']+'.wav'), axis=1)
            data_list=pd.concat([data_list,trans])
    data_list.reset_index(drop=True,inplace=True)

    return data_list

def get_padded_melspec(audio, sample_rate):
    audio=audio[0].unsqueeze(0)

    if sample_rate!=config.sample_rate:
        audio=torchaudio.transforms.Resample(sample_rate, config.sample_rate)(audio)
    melspec=utils.extract_normed_mel(audio)

    padding_len=32-melspec.shape[-1]%32
    if padding_len==0:
        padding_len=32
    melspec=torch.nn.functional.pad(melspec,(0,padding_len))
    if len(melspec.shape)>3:
        melspec=melspec.squeeze(0)
    melspec=melspec.squeeze(0).numpy().astype('float32')

    return melspec

def full_label_binarize(data_list,name='train'):
    idx_data=[]
    data_file=open(os.path.join('data','full_label',name+'.data'), 'wb')

    if config.wavlm.enabled:
        checkpoint = torch.load(config.wavlm.path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model=model.to(config.device)
        model.eval()
    
    for index in trange(len(data_list)):
        meta_data={}
        # return: input_feature, seg_target
        # melspec
        audio, sample_rate = torchaudio.load(data_list.iloc[index]['path'])
        audio=audio[0].unsqueeze(0)

        if sample_rate!=config.sample_rate:
            audio=torchaudio.transforms.Resample(sample_rate, config.sample_rate)(audio)
        melspec=utils.extract_normed_mel(audio)

        padding_len=32-melspec.shape[-1]%32
        if padding_len==0:
            padding_len=32
        melspec=torch.nn.functional.pad(melspec,(0,padding_len))
        if len(melspec.shape)>3:
            melspec=melspec.squeeze(0)
        melspec=melspec.squeeze(0).numpy().astype('float32')

        T=melspec.shape[-1]

        if config.wavlm.enabled:
            # extract the representation of last layer
            wav_input_16khz = torchaudio.transforms.Resample(config.sample_rate,16000)(audio)
            wav_input_16khz = torch.nn.functional.pad(wav_input_16khz,(0,80))
            if cfg.normalize:
                wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
            rep = model.extract_features(wav_input_16khz.to(config.device))[0]
            rep = repeat(rep,'B T C -> B C (T T2)',T2=2)
            if rep.shape[-1]>melspec.shape[-1]:
                rep=rep[:,:,:melspec.shape[-1]]
            elif rep.shape[-1]<melspec.shape[-1]:
                rep=torch.nn.functional.pad(rep,(0,melspec.shape[-1]-rep.shape[-1]))
            rep=rep.squeeze(0).detach().cpu().numpy().astype('float32')

            input_feature=np.concatenate([melspec,rep],axis=0)
        else:
            input_feature=melspec
        assert(len(input_feature.shape)==2)
        
        meta_data['input_feature']={}
        wirte_ndarray_to_bin(data_file,meta_data['input_feature'],input_feature)

        # ctc_target
        ph_seq=[i for i in data_list.iloc[index]['ph_seq'].split(' ') if i !='']
        ph_seq_num=[vocab[ph] for ph in ph_seq]
        # ctc_target=np.array(ph_seq_num)
        # ctc_target=ctc_target.astype('int32')

        # meta_data['ctc_target']={}
        # wirte_ndarray_to_bin(data_file,meta_data['ctc_target'],ctc_target)

        # seg_target
        ph_dur=torch.tensor([float(i) for i in data_list.iloc[index]['ph_dur'].split(' ') if i !=''])
        if (len(ph_dur)!=len(ph_seq_num)):
            print(data_list.iloc[index]['path'],len(ph_dur),len(ph_seq_num))
        assert(len(ph_dur)==len(ph_seq_num))
        ph_time=ph_dur.cumsum(dim=0)*config.sample_rate/config.hop_length
        ph_time_int=(ph_time).round().int()
        ph_time_diff_with_int=ph_time-ph_time_int
        ph_time_int=torch.cat([torch.tensor([0]),ph_time_int])
        target=torch.zeros(T)
        for i in range(len(ph_seq_num)):
            target[ph_time_int[i]:ph_time_int[i+1]]=ph_seq_num[i]
        
        seg_target=target.numpy().astype('int32')
        seg_target=np.expand_dims(seg_target,0)

        meta_data['seg_target']={}
        wirte_ndarray_to_bin(data_file,meta_data['seg_target'],seg_target)


        # edge_target
        edge_target=np.zeros_like(seg_target[0])+config.label_smoothing/(1-config.label_smoothing)
        for i in range(len(ph_seq_num)-1):
            if not(ph_seq_num[i]==0 and ph_seq_num[i+1]==0):
                if ph_time_int[i+1]==0 or ph_time_int[i+1]==len(ph_seq_num)-1:
                    edge_target[ph_time_int[i+1]]=1
                else:
                    edge_target[ph_time_int[i+1]]=0.5-ph_time_diff_with_int[i]
                    edge_target[ph_time_int[i+1]-1]=0.5+ph_time_diff_with_int[i]

        edge_target=edge_target.astype('float32')*(1-config.label_smoothing)
        edge_target=np.array([edge_target,1-edge_target])

        meta_data['edge_target']={}
        wirte_ndarray_to_bin(data_file,meta_data['edge_target'],edge_target)


        idx_data.append(meta_data)

    data_file.close()
    idx_data=pd.DataFrame(idx_data)
    idx_data.to_pickle(os.path.join('data','full_label',name+'.idx'))

# no label
def no_label_binarize(name='train'):
    file_path_list=[]
    for path,folders,files in os.walk(os.path.join('data')):
        for file in files:
            if file.endswith('.wav'):
                file_path_list.append(os.path.join(path,file))

    idx_data=[]
    data_file=open(os.path.join('data','no_label',name+'.data'), 'wb')
    audio_aug=AudioAugmentation()

    for index,path in enumerate(tqdm(file_path_list)):
        # return: input_feature, input_feature_weak_aug, input_feature_strong_aug

        meta_data={}
        if config.wavlm.enabled:
            pass
        
        audio, sample_rate = torchaudio.load(file_path_list[index])
        melspec=get_padded_melspec(audio, sample_rate)

        if config.wavlm.enabled:
            pass
        else:
            input_feature=melspec


        meta_data['input_feature']={}
        wirte_ndarray_to_bin(data_file,meta_data['input_feature'],input_feature)


        audio_weak_aug,audio_strong_aug=audio_aug(audio)
        melspec_weak_aug=get_padded_melspec(audio_weak_aug, sample_rate)

        if config.wavlm.enabled:
            pass
        else:
            input_feature_weak_aug=melspec_weak_aug

        meta_data['input_feature_weak_aug']={}
        wirte_ndarray_to_bin(data_file,meta_data['input_feature_weak_aug'],input_feature_weak_aug)


        melspec_strong_aug=get_padded_melspec(audio_strong_aug, sample_rate)

        if config.wavlm.enabled:
            pass
        else:
            input_feature_strong_aug=melspec_strong_aug

        meta_data['input_feature_strong_aug']={}
        wirte_ndarray_to_bin(data_file,meta_data['input_feature_strong_aug'],input_feature_strong_aug)

        idx_data.append(meta_data)


    data_file.close()
    idx_data=pd.DataFrame(idx_data)
    idx_data.to_pickle(os.path.join('data','no_label',name+'.idx'))


if __name__=='__main__':
    data_list=get_data_list('full_label')
    valid_list_length=int(config.valid_length)
    valid_list=data_list.sample(valid_list_length)
    train_list=data_list.drop(valid_list.index)

    full_label_binarize(valid_list,'valid')
    full_label_binarize(train_list,'train')

    # no_label_binarize('train')
