import random

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, Sampler


class MixedDataset(Dataset):
    def __init__(self, df):
        df.reset_index(drop=True, inplace=True)
        self.df = df

    def get_wav_lengths(self):
        return self.df["wav_length"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # wav, normal_id_seq, normal_interval_seq, special_id_seq, special_interval_seq, wav_length, label_type
        row = self.df.iloc[index]

        wav, _ = torchaudio.load(str(row["wav_path"]))
        wav = wav.squeeze(0)
        normal_id_seq = row["normal_id_seq"]
        normal_interval_seq = row["normal_interval_seq"]
        special_id_seq = row["special_id_seq"]
        special_interval_seq = row["special_interval_seq"]
        # wav_length = row["wav_length"]
        label_type = row["label_type"]

        return (
            wav,
            normal_id_seq,
            normal_interval_seq,
            special_id_seq,
            special_interval_seq,
            # wav_length,
            label_type,
        )


class SortedLengthRandomizedBatchSampler(Sampler):
    """
    按照音频长度顺序划分batch，batch的每个sample总长度差不多，所需的padding少，算力浪费少；
    每个epoch前，音频长度加上一个随机数再进行排序，增加随机性；
    第一个epoch的batch按照sample长度升序排序，相当于使用课程学习降低训练难度。
    """

    def __init__(
        self,
        wav_lengths: pd.Series,
        batch_length: float = 100,  # unit: second
        randomize_factor: float = 0.2,
    ):
        self.wav_lengths = wav_lengths.sort_values(ascending=False)
        self.batch_length = batch_length
        self.randomize_factor = randomize_factor

        # 扔掉音频长度大于batch_length的样本
        self.wav_lengths = self.wav_lengths[self.wav_lengths <= self.batch_length]

        self.is_first_epoch = None
        self.len = len(self.get_batches())
        # len不固定，根据随机数的不同会有细微的变化
        # 如果多了就随机扔掉一点，少了就过采样一点

    def get_batches(self):
        randomized_wav_lengths = self.wav_lengths.map(
            lambda x: x * (1 + np.random.rand() * self.randomize_factor)
        )
        randomized_wav_lengths = randomized_wav_lengths.sort_values()
        wav_lengths = self.wav_lengths[randomized_wav_lengths.index]

        batches = []

        curr_start_idx = 0
        while curr_start_idx < len(wav_lengths):
            curr_batch_max_length = wav_lengths[curr_start_idx]
            curr_batch_size = int(self.batch_length // curr_batch_max_length)
            # curr_batch_size必定大于等于1，因为之前扔掉了太长的sample

            batches.append(
                wav_lengths.index[curr_start_idx : curr_start_idx + curr_batch_size]
            )

            curr_start_idx = curr_start_idx + curr_batch_size

        if self.is_first_epoch is True:
            self.is_first_epoch = False
            return batches[::-1]
        elif self.is_first_epoch is None:
            self.is_first_epoch = True

        random.shuffle(batches)
        return batches

    def __len__(self):
        return self.len

    def __iter__(self):
        batches = self.get_batches()
        if len(batches) > self.len:
            indices = random.sample(range(len(batches)), self.len)
            indices.sort()
            batches = [batches[i] for i in indices]
        elif len(batches) < self.len:
            extra_indices = random.sample(range(len(batches)), self.len - len(batches))
            indices = list(range(len(batches)))
            indices.extend(extra_indices)
            batches = [batches[i] for i in indices]
        return iter(batches)


def collate_fn(batch):
    """
    输入：
        wav (T)
        normal_id_seq (L)
        normal_interval_seq (L, 2)
        special_id_seq (L1)
        special_interval_seq (L1, 2)
        label_type (int)
    返回:
        audios (B, max_T)
        audio_lengths (B)
        normal_id_seqs (num_weaks, max_L)
        normal_interval_seqs (num_full_labels, max_L, 2)
        normal_id_lengths (num_weaks)
        special_id_seqs (num_weaks, max_L1)
        special_interval_seqs (num_full_labels, max_L1, 2)
        special_id_lengths (num_weaks)
        label_types (B)
    """
    (
        audios,
        normal_id_seqs,
        normal_interval_seqs,
        special_id_seqs,
        special_interval_seqs,
        label_types,
    ) = list(zip(*batch))

    normal_id_seqs = [torch.tensor(n) for n in normal_id_seqs if n is not None]
    normal_interval_seqs = [
        torch.tensor(n) for n in normal_interval_seqs if n is not None
    ]
    special_id_seqs = [torch.tensor(s) for s in special_id_seqs if s is not None]
    special_interval_seqs = [
        torch.tensor(s) for s in special_interval_seqs if s is not None
    ]

    audio_lengths = torch.tensor([len(a) for a in audios])
    normal_id_lengths = torch.tensor([len(n) for n in normal_id_seqs])
    special_id_lengths = torch.tensor([len(s) for s in special_id_seqs])

    # padding
    audios = nn.utils.rnn.pad_sequence(audios, batch_first=True)
    normal_id_seqs = (
        nn.utils.rnn.pad_sequence(normal_id_seqs, batch_first=True)
        if len(normal_id_seqs) > 0
        else torch.empty(0, 0)
    )
    normal_interval_seqs = (
        nn.utils.rnn.pad_sequence(normal_interval_seqs, batch_first=True)
        if len(normal_interval_seqs) > 0
        else torch.empty(0, 0, 2)
    )
    special_id_seqs = (
        nn.utils.rnn.pad_sequence(special_id_seqs, batch_first=True)
        if len(special_id_seqs) > 0
        else torch.empty(0, 0)
    )
    special_interval_seqs = (
        nn.utils.rnn.pad_sequence(special_interval_seqs, batch_first=True)
        if len(special_interval_seqs) > 0
        else torch.empty(0, 0, 2)
    )

    label_types = torch.tensor(label_types)

    return (
        audios,
        audio_lengths,
        normal_id_seqs,
        normal_interval_seqs,
        normal_id_lengths,
        special_id_seqs,
        special_interval_seqs,
        special_id_lengths,
        label_types,
    )
