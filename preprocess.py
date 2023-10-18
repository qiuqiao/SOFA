import torch
import pandas as pd
import numpy as np
import utils
from utils import dict_to_namespace
from utils import wirte_ndarray_to_bin
import os
import yaml
from tqdm import trange
import os
import warnings
import argparse

def generate_vocab():
    print('generating vocabulary...')
    phonemes=[]
    for path,folders,files in os.walk('data'):
        if 'transcriptions.csv' in files:
            df = pd.read_csv(os.path.join(path,'transcriptions.csv'))
            ph=sorted(set(' '.join(df['ph_seq']).split(' ')))
            phonemes.extend(ph)

    phonemes=set(phonemes)
    for p in config.ignore_phonemes:
        if p in phonemes:
            phonemes.remove(p)
    phonemes=sorted(phonemes)
    phonemes=['<EMPTY>',*phonemes]

    vocab={phonemes[i]:i for i in range(len(phonemes))}
    vocab.update({i:phonemes[i] for i in range(len(phonemes))})
    vocab.update({p:0 for p in config.ignore_phonemes})
    vocab.update({'<vocab_size>':len(phonemes)})

    with open(os.path.join('ckpt',config.model_name,'vocab.yaml'), 'w') as file:
        yaml.dump(vocab, file)
    print('done.')

    return vocab

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


def full_label_binarize(data_list,name='train'):
    idx_data=[]
    data_file=open(os.path.join('data','full_label',name+'.data'), 'wb')
    
    for index in trange(len(data_list)):
        meta_data={}
        # return: input_feature, seg_target
        # melspec
        audio=utils.load_resampled_audio(data_list.iloc[index]['path'])
        melspec=utils.get_padded_melspec(audio)
        if len(melspec.shape)>2:
            melspec=melspec.squeeze(0)
        T=melspec.shape[-1]

        if T>config.melspec_maxlength:
            warnings.warn(f'Melspec of {data_list.iloc[index]["path"][:-4]} has a length of {T}, which is longer than {config.melspec_maxlength}. Ignored.')
            continue

        input_feature=melspec
        input_feature=input_feature.cpu().numpy().astype('float32')
        assert(len(input_feature.shape)==2)
        
        meta_data['input_feature']={}
        wirte_ndarray_to_bin(data_file,meta_data['input_feature'],input_feature)

        # seg_target
        ph_seq=[i for i in data_list.iloc[index]['ph_seq'].split(' ') if i !='']
        ph_seq_num=[vocab[ph] for ph in ph_seq]

        ph_dur=torch.tensor([float(i) for i in data_list.iloc[index]['ph_dur'].split(' ') if i !=''])
        assert(len(ph_dur)==len(ph_seq_num))
        ph_time=ph_dur.cumsum(dim=0)*config.sample_rate/config.hop_length
        ph_time_int=(ph_time).round().int()
        ph_time_diff_with_int=ph_time-ph_time_int
        ph_time_int=torch.cat([torch.tensor([0]),ph_time_int])
        # ph_time_int[ph_time_int>=T]=T-1
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


def weak_label_binarize(data_list,name='train'):
    idx_data=[]
    data_file=open(os.path.join('data','weak_label',name+'.data'), 'wb')
    
    for index in trange(len(data_list)):
        meta_data={}
        # return: input_feature, ctc_target
        # melspec
        audio=utils.load_resampled_audio(data_list.iloc[index]['path'])
        melspec=utils.get_padded_melspec(audio)
        melspec=melspec.squeeze(0).cpu().numpy().astype('float32')

        T=melspec.shape[-1]
        if T>config.melspec_maxlength:
            warnings.warn(f'Melspec of {data_list.iloc[index]["path"][:-4]} has a length of {T}, which is longer than {config.melspec_maxlength}. Ignored.')
            continue

        input_feature=melspec
        meta_data['input_feature']={}
        wirte_ndarray_to_bin(data_file,meta_data['input_feature'],input_feature)

        # ctc_target
        ph_seq=[i for i in data_list.iloc[index]['ph_seq'].split(' ') if i !='']
        ph_seq_num=[]
        for ph in ph_seq:
            assert ph in vocab,f'"{ph}" from {data_list.iloc[index]["path"][:-4]} is not in vocab. please check your dataset or rerun "vocab_gen.py".'
            if vocab[ph]!=0:
                ph_seq_num.append(vocab[ph])
        ctc_target=np.array(ph_seq_num)
        ctc_target=ctc_target.astype('int32')

        meta_data['ctc_target']={}
        wirte_ndarray_to_bin(data_file,meta_data['ctc_target'],ctc_target)


        idx_data.append(meta_data)

    data_file.close()
    idx_data=pd.DataFrame(idx_data)
    idx_data.to_pickle(os.path.join('data','weak_label',name+'.idx'))

def parse_args():
    description='Preprocess the dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c','--config_path', type=str, default=os.path.join('configs','config.yaml'), help='Path to config file.')

    return parser.parse_args()

def get_config():
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    config=dict_to_namespace(config)

    return config

def copy_config():
    with open(args.config_path, 'r') as file:
        config_text=file.read()
    if not os.path.exists(os.path.join('ckpt',config.model_name)):
        os.makedirs(os.path.join('ckpt',config.model_name))
    with open(os.path.join('ckpt',config.model_name,'config.yaml'), 'w') as file:
        file.write(config_text)

if __name__=='__main__':
    args=parse_args()
    config=get_config()
    utils.init_config(config)# temporary solution
    copy_config()

    vocab=generate_vocab()

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    data_list=get_data_list('full_label')
    valid_list_length=int(config.valid_length)
    valid_list=data_list.sample(valid_list_length)
    train_list=data_list.drop(valid_list.index)
    full_label_binarize(valid_list,'valid')
    full_label_binarize(train_list,'train')

    data_list=pd.concat([get_data_list('full_label'),get_data_list('weak_label')])
    data_list.reset_index(drop=True,inplace=True)
    valid_list_length=int(config.valid_length)
    valid_list=data_list.sample(valid_list_length)
    train_list=data_list.drop(valid_list.index)
    weak_label_binarize(valid_list,'valid')
    weak_label_binarize(train_list,'train')