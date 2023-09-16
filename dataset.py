import torch
import torchaudio
import pandas as pd
import utils
import os
import yaml
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class FullLabelDataset(torch.utils.data.Dataset):
    # return: 
    #   tensor[C,T], input_feature
    #   tensor[T], target with class indices
    def __init__(self,name):
        self.idx_data=pd.read_pickle(os.path.join('data','full_label',name+'.idx'))
        self.data_file=open(os.path.join('data','full_label',name+'.data'), 'rb')

    def __len__(self):
        return len(self.idx_data)

    def __getitem__(self, index):
        # input_feature
        input_feature=utils.read_ndarray_from_bin(self.data_file,self.idx_data['input_feature'][index])
        input_feature=torch.tensor(input_feature).float()

        # seg_target
        seg_target=utils.read_ndarray_from_bin(self.data_file,self.idx_data['seg_target'][index])
        seg_target=torch.tensor(seg_target).long()

        # edge_target
        edge_target=utils.read_ndarray_from_bin(self.data_file,self.idx_data['edge_target'][index])
        edge_target=torch.tensor(edge_target).float()

        return input_feature,seg_target,edge_target


def collate_fn(batch):
    max_len=[]
    for param in range(len(batch[0])):
        max_len.append(max([i[param].shape[-1] for i in batch]))

    for i in range(len(batch)):
        batch_list = list(batch[i])
        for param in range(len(max_len)):
            # batch_list[param] = np.pad(batch_list[param], ((0, 0),(0, max_len[param] - batch_list[param].shape[-1])), 'constant')
            batch_list[param] = torch.nn.functional.pad(torch.tensor(batch_list[param]), (0, max_len[param] - batch_list[param].shape[-1]), 'constant', 0)
        batch[i] = tuple(batch_list)
    
    res=[]
    for param in range(len(batch[0])):
        # res.append(np.stack([i[param] for i in batch]))
        res.append(torch.stack([i[param] for i in batch]))

    return tuple(res)


class NoLabelDataset(torch.utils.data.Dataset):
    def __init__(self,name):
        self.idx_data=pd.read_pickle(os.path.join('data','no_label',name+'.idx'))
        self.data_file=open(os.path.join('data','no_label',name+'.data'), 'rb')

    def __len__(self):
        return len(self.idx_data)

    def __getitem__(self, index):
        # input_feature
        input_feature=utils.read_ndarray_from_bin(self.data_file,self.idx_data['input_feature'][index])
        input_feature=torch.tensor(input_feature).float()

        # input_feature_weak_aug
        input_feature_weak_aug=utils.read_ndarray_from_bin(self.data_file,self.idx_data['input_feature_weak_aug'][index])
        input_feature_weak_aug=torch.tensor(input_feature_weak_aug).float()

        # input_feature_strong_aug
        input_feature_strong_aug=utils.read_ndarray_from_bin(self.data_file,self.idx_data['input_feature_strong_aug'][index])
        input_feature_strong_aug=torch.tensor(input_feature_strong_aug).float()

        return input_feature,input_feature_weak_aug,input_feature_strong_aug

