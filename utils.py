import matplotlib.pyplot as plt
import yaml
import numpy as np
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import pickle
from argparse import Namespace
import torch.nn as nn

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

extract_mel = T.MelSpectrogram(
    sample_rate=config['sample_rate'],
    n_fft=config['n_fft'],
    win_length=None,
    hop_length=config['hop_length'],
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=config['n_mels'],
    mel_scale="htk",
).float()

def extract_normed_mel(waveform):
    melspec = extract_mel(waveform.float()).unsqueeze(0)
    melspec = F.amplitude_to_DB(melspec, multiplier=10., amin=1e-10,db_multiplier=1.0, top_db=90.0)
    melspec= (melspec-torch.mean(melspec))/(torch.std(melspec))
    if len(melspec.shape)>3:
        melspec=melspec.squeeze(0)
    return melspec.float()

def dict_to_namespace(d):
    namespace = Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def plot_spectrogram_and_phonemes(specgram, ph_seq=None, ph_dur=None, target_gt=None, target_pred=None, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or " ")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow((specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)

    if ph_seq is not None and ph_dur is not None:
        ph_time=np.cumsum(np.array(ph_dur))*config['sample_rate']/config['hop_length']
        ph_time=np.insert(ph_time, 0, 0)
        for i in ph_time:
            plt.axvline(i,color='r',linewidth=1)
        for i in range(0,len(ph_seq),2):
            if ph_seq[i]!='<EMPTY>':
                plt.text(ph_time[i]-ph_dur[i],len(specgram),ph_seq[i],fontsize=11)
        for i in range(1,len(ph_seq),2):
            if ph_seq[i]!='<EMPTY>':
                plt.text(ph_time[i],len(specgram)-5,ph_seq[i],fontsize=11,color='white')

    if target_pred is not None:
        target_pred_locations = np.arange(len(target_pred))
        plt.plot(target_pred_locations, target_pred, color='r', linewidth=0.5)
    
    if target_gt is not None:
        target_gt_locations = np.arange(len(target_gt))
        plt.plot(target_gt_locations, target_gt, color='black', linewidth=1, alpha=0.6)
        plt.fill_between(target_gt_locations, target_gt, color='black', alpha=0.3)

    fig.set_size_inches(15,5)

    return fig

class SinScheduler():
    def __init__(self, max_steps, start_steps, end_steps):
        self.max_steps=max_steps
        self.start_steps=start_steps
        self.end_steps=end_steps
        self.curr_steps=0
    
    def __call__(self):
        if self.curr_steps<self.start_steps:
            return 0
        elif self.curr_steps<self.end_steps:
            return np.sin(-np.pi/2+np.pi*(self.curr_steps-self.start_steps)/(self.end_steps-self.start_steps))/2+0.5
        else:
            return 1
    
    def step(self):
        self.curr_steps+=1

class GaussianRampUpScheduler():
    def __init__(self, max_steps, start_steps, end_steps):
        self.max_steps=max_steps
        self.start_steps=start_steps
        self.end_steps=end_steps
        self.curr_steps=0
    
    def __call__(self):
        if self.curr_steps<self.start_steps:
            return 0
        elif self.curr_steps<self.end_steps:
            return np.exp(-5*(1-(self.curr_steps-self.start_steps)/(self.end_steps-self.start_steps))**2)
        else:
            return 1
    
    def step(self):
        self.curr_steps+=1


def wirte_ndarray_to_bin(file,idx_data,array):
    idx_data['start']=file.tell()
    idx_data['shape']=array.shape
    idx_data['dtype']=str(array.dtype)

    array=array.tobytes()
    file.write(array)
    idx_data['len']=file.tell()-idx_data['start']

def read_ndarray_from_bin(file,idx_data):
    file.seek(idx_data['start'],0)
    return np.frombuffer(file.read(idx_data['len']),dtype=idx_data['dtype']).reshape(idx_data['shape'])

def save_dict(dict_obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_obj, file)

def load_dict(file_path):
    with open(file_path, 'rb') as file:
        dict_obj = pickle.load(file)
    return dict_obj

class BinaryEMDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=torch.nn.L1Loss()
    
    def forward(self, pred, target):
        # pred, target: [B,T]
        loss=self.loss(pred.cumsum(dim=-1), target.cumsum(dim=-1))
        loss+=self.loss(pred.flip([-1]).cumsum(dim=-1), target.flip([-1]).cumsum(dim=-1))
        return loss/2

class GHMLoss(torch.nn.Module):
    def __init__(self, num_classes,num_prob_bins=10,alpha=0.99,label_smoothing=0.0,enable_prob_input=False):
        super().__init__()
        self.enable_prob_input=enable_prob_input
        self.num_classes=num_classes
        self.num_prob_bins=num_prob_bins
        if not enable_prob_input:
            self.classes_ema=torch.ones(num_classes).to(config['device'])
        self.prob_bins_ema=torch.ones(num_prob_bins).to(config['device'])
        self.alpha=alpha
        self.loss_fn=nn.CrossEntropyLoss(reduction='none',label_smoothing=label_smoothing)
    
    def forward(self, pred, target):

        pred_prob=torch.softmax(pred,dim=1)
        if not self.enable_prob_input:
            target_prob=torch.zeros_like(pred_prob).scatter_(1,target.unsqueeze(1),1).to(config['device'])
        else:
            target_prob=target
        pred_prob=(pred_prob*target_prob).sum(dim=1).clamp(1e-6,1-1e-6)

        loss=self.loss_fn(pred,target)
        if not self.enable_prob_input:
            loss_classes=target.long()
            # print(len(self.classes_ema),loss_classes.min().cpu().numpy(),loss_classes.max().cpu().numpy(),len(self.prob_bins_ema),torch.floor(pred_prob*self.num_prob_bins).long().min().cpu().numpy(),torch.floor(pred_prob*self.num_prob_bins-1e-10).long().max().cpu().numpy())
            loss_weighted=loss/torch.sqrt((self.classes_ema[loss_classes]*self.prob_bins_ema[torch.floor(pred_prob*self.num_prob_bins-1e-6).long()]+1e-10))
        else:
            loss_weighted=loss/(self.prob_bins_ema[torch.floor(pred_prob*self.num_prob_bins).long()]+1e-10)
        loss=torch.mean(loss_weighted)

        prob_bins=torch.histc(pred_prob,bins=self.num_prob_bins,min=0,max=1).to(config['device'])
        prob_bins=prob_bins/(torch.sum(prob_bins)+1e-10)*self.num_prob_bins
        self.prob_bins_ema=self.prob_bins_ema*self.alpha+(1-self.alpha)*prob_bins
        self.prob_bins_ema=self.prob_bins_ema/(torch.sum(self.prob_bins_ema)+1e-10)*self.num_prob_bins

        if not self.enable_prob_input:
            classes=torch.histc(target.float(),bins=self.num_classes,min=0,max=self.num_classes-1).to(config['device'])
            classes=classes/(torch.sum(classes)+1e-10)*self.num_classes
            self.classes_ema=self.classes_ema*self.alpha+(1-self.alpha)*classes
            self.classes_ema=self.classes_ema/(torch.sum(self.classes_ema)+1e-10)*self.num_classes

        return loss


def confusion_matrix(num_classes,y_true,y_predict):
    y_true,y_predict=y_true.cpu(),y_predict.cpu()
    matrix = np.zeros((num_classes,num_classes))
    for i in range(len(y_true)):
        matrix[y_true[i],y_predict[i]] += 1
    return matrix

def cal_accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def cal_precision(confusion_matrix):
    precision = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        precision[i] = confusion_matrix[i,i] / (confusion_matrix[:,i].sum()+1e-10)
    return precision

def cal_recall(confusion_matrix):
    recall = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        recall[i] = confusion_matrix[i,i] / (confusion_matrix[i,:].sum()+1e-10)
    return recall

def cal_macro_F1(confusion_matrix):
    precision = cal_precision(confusion_matrix)
    recall = cal_recall(confusion_matrix)
    return 2 * precision * recall / (precision + recall+1e-10)

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im)
    
    return fig