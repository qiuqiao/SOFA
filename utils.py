import matplotlib.pyplot as plt
import yaml
import librosa
import numpy as np
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import pickle
from argparse import Namespace

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
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow((specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)

    if ph_seq is not None and ph_dur is not None:
        line_locations=np.cumsum(np.array(ph_dur))*config['sample_rate']/config['hop_length']
        for i in line_locations:
            plt.axvline(i,color='r')
        for i in range(len(ph_seq)):
            if ph_seq[i]!='<EMPTY>':
                plt.text(line_locations[i]-ph_dur[i]*config['sample_rate']/config['hop_length']/2,0,ph_seq[i],fontsize=12)

    if target_pred is not None:
        target_pred_locations = np.arange(len(target_pred))
        plt.plot(target_pred_locations, target_pred, color='r', linewidth=2)
    
    if target_gt is not None:
        target_gt_locations = np.arange(len(target_gt))
        plt.plot(target_gt_locations, target_gt, color='black', linewidth=2)

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
        loss=self.loss(pred.cumsum(dim=-1), target.cumsum(dim=-1))/target.shape[-1]
        loss+=self.loss(pred.flip([-1]).cumsum(dim=-1), target.flip([-1]).cumsum(dim=-1))/target.shape[-1]
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