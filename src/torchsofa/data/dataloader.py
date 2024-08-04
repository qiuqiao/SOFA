import random

import numpy as np
import pandas as pd
import torchaudio
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
        # wav, normal_id_seq, normal_inverval_seq, special_id_seq, special_inverval_seq, wav_length, label_type
        row = self.df.iloc[index]

        wav, _ = torchaudio.load(row["wav_path"])
        normal_id_seq = row["normal_id_seq"]
        normal_inverval_seq = row["normal_inverval_seq"]
        special_id_seq = row["special_id_seq"]
        special_inverval_seq = row["special_inverval_seq"]
        wav_length = row["wav_length"]
        label_type = row["label_type"]

        return (
            wav,
            normal_id_seq,
            normal_inverval_seq,
            special_id_seq,
            special_inverval_seq,
            wav_length,
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

        self.len = len(self.get_batches())
        # len不固定，根据随机数的不同会有细微的变化
        # 如果多了就随机扔掉一点，少了就过采样一点
        self.is_first_epoch = True

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
            curr_batch_size = self.batch_length // curr_batch_max_length
            # curr_batch_size必定大于等于1，因为之前扔掉了太长的sample

            batches.append(
                wav_lengths.index[curr_start_idx : curr_start_idx + curr_batch_size]
            )

            curr_start_idx = curr_start_idx + curr_batch_size

        if self.is_first_epoch:
            self.is_first_epoch = False
            return batches[::-1]

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


# def collate_fn(batch):
#     """
#     一个batch里有：wav, normal_id_seq, normal_inverval_seq, special_id_seq, special_inverval_seq, wav_length, label_type
#     返回： wav, normal_id_seq, normal_inverval_seq, normal_id_lengths, special_id_seq, special_inverval_seq, special_id_lengths, wav_length, label_type
#     """
