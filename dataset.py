import torch
import pandas as pd
import utils
import os
import pathlib
import numpy as np


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, binary_data_folder, prefix):
        self.idx_data = pd.read_pickle(
            pathlib.Path(binary_data_folder) / (prefix + ".idx")
        )
        self.data_file = open(
            pathlib.Path(binary_data_folder) / (prefix + ".data"), "rb"
        )

    def __len__(self):
        return len(self.idx_data)

    def read_ndarray_from_bin(self, file, idx_data):
        file.seek(idx_data["start"], 0)
        return np.frombuffer(
            file.read(idx_data["len"]), dtype=idx_data["dtype"]
        ).reshape(idx_data["shape"])

    def __getitem__(self, index):
        # input_feature
        input_feature = self.read_ndarray_from_bin(
            self.data_file, self.idx_data["input_feature"][index]
        )
        input_feature = torch.tensor(input_feature).float()

        # ph_seq
        ph_seq = self.read_ndarray_from_bin(
            self.data_file, self.idx_data["ph_seq"][index]
        )
        ph_seq = torch.tensor(ph_seq).long()

        # ph_edge
        ph_edge = self.read_ndarray_from_bin(
            self.data_file, self.idx_data["ph_edge"][index]
        )
        ph_edge = torch.tensor(ph_edge).long()

        # ph_frame
        ph_frame = self.read_ndarray_from_bin(
            self.data_file, self.idx_data["ph_frame"][index]
        )
        ph_frame = torch.tensor(ph_frame).float()

        return input_feature, ph_seq, ph_edge, ph_frame


def collate_fn(batch):
    max_len = [
        max([i[param].shape[-1] for i in batch]) for param in range(len(batch[0]))
    ]

    for i, item in enumerate(batch):
        item = list(item)
        item[1] = torch.tensor(item[1]).long()
        for param in [0, 2, 3]:
            item[param] = torch.nn.functional.pad(
                torch.tensor(item[param]),
                (0, max_len[param] - item[param].shape[-1]),
                "constant",
                0,
            )
        batch[i] = tuple(item)

    ph_seq = torch.cat([item[1] for item in batch])
    ph_seq_lengths = torch.tensor([len(item[1]) for item in batch])

    input_feature = torch.stack([item[0] for item in batch])
    ph_edge = torch.stack([item[2] for item in batch])
    ph_frame = torch.stack([item[3] for item in batch])

    return (
        input_feature,
        ph_seq,
        ph_seq_lengths,
        ph_edge,
        ph_frame,
    )
