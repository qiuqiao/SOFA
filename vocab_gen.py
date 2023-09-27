import os
import pandas as pd
import yaml
import utils
from argparse import Namespace

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config=utils.dict_to_namespace(config)

with open('vocab.yaml', 'r') as file:
    dict = yaml.safe_load(file)


phonemes=[]
for path,folders,files in os.walk('data'):
    if 'transcriptions.csv' in files:
        df = pd.read_csv(os.path.join(path,'transcriptions.csv'))
        ph=sorted(set(' '.join(df['ph_seq']).split(' ')))
        phonemes.extend(ph)
        if 'iou' in ph:
            print(os.path.join(path,'transcriptions.csv'))

phonemes=set(phonemes)
for p in config.special_phonemes:
    if p in phonemes:
        phonemes.remove(p)
phonemes=sorted(phonemes)
phonemes=['<EMPTY>',*phonemes]

vocab={phonemes[i]:i for i in range(len(phonemes))}
vocab.update({i:phonemes[i] for i in range(len(phonemes))})
vocab.update({p:0 for p in config.special_phonemes})
vocab.update({'<vocab_size>':len(phonemes)})

with open('vocab.yaml', 'w') as file:
    yaml.dump(vocab, file)