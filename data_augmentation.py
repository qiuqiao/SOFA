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
import random

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
config

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)


class AudioAugmentation():
    def __init__(self):
        self.weak_effect_groups = [
            [
                ['equalizer','800','1q','12'],
                ['treble','10'],
                ['bass','15'],
            ],
            [
                ['pitch','80'],
                ['pitch','-80'],
                ['tremolo','3','20'],
                ['tremolo','5','15'],
            ],
            [
                ['rate',f'{config.sample_rate}'],
            ]
        ]

        self.strong_effect_groups = [
            [
                ['equalizer','800','1q','-15'],
                ["highpass", "-2", "300"],
                ['lowpass','-1','1500'],
            ],
            [
                ['pitch','160'],
                ['pitch','-160'],
                ['tremolo','4','40'],
                ['tremolo','6','30'],
            ],
            [
                ['rate',f'{config.sample_rate}'],
            ]
        ]
    
    def __call__(self, x):
        x_peak=torch.max(torch.abs(x))
        x=x*0.08/x_peak

        weak_effects=[]
        for i in self.weak_effect_groups:
            weak_effects.append(random.choice(i))
        x_weak, sample_rate_weak = torchaudio.sox_effects.apply_effects_tensor(x, config.sample_rate, weak_effects)
        x_weak_peak=torch.max(torch.abs(x_weak))
        x_weak=x_weak*0.08/x_weak_peak

        strong_effects=[]
        for i in self.strong_effect_groups:
            strong_effects.append(random.choice(i))
        x_strong, sample_rate_strong = torchaudio.sox_effects.apply_effects_tensor(x, config.sample_rate, strong_effects)
        x_strong_peak=torch.max(torch.abs(x_strong))
        x_strong=x_strong*0.08/x_strong_peak

        assert(sample_rate_weak==config.sample_rate and sample_rate_strong==config.sample_rate)

        return x_weak, x_strong
