import torch
import h5py
import numpy as np
import pathlib


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, binary_data_folder="data/binary", prefix="train"):
        # do not open hdf5 here
        self.h5py_file = None
        self.h5py_items = None
        self.label_type_ids = None
        self.keys = None
        self.binary_data_folder = binary_data_folder
        self.prefix = prefix

    def open_h5py_file(self):
        self.h5py_file = h5py.File(str(pathlib.Path(self.binary_data_folder) / (self.prefix + ".h5py")), 'r')
        self.h5py_items = self.h5py_file["items"]
        self.label_type_ids = np.array(self.h5py_file["label_type_ids"])
        self.keys = list(self.h5py_items.keys())

    def __len__(self):
        if self.h5py_items is None:
            self.open_h5py_file()
        return len(self.keys)

    def __getitem__(self, index):
        if self.h5py_items is None:
            self.open_h5py_file()

        item = self.h5py_items[self.keys[index]]

        # input_feature
        input_feature = np.array(item["input_feature"])

        # label_type
        label_type = np.array(item["label_type"])

        # ph_seq
        ph_seq = np.array(item["ph_seq"])

        # ph_edge
        ph_edge = np.array(item["ph_edge"])

        # ph_frame
        ph_frame = np.array(item["ph_frame"])

        return input_feature, ph_seq, ph_edge, ph_frame, label_type


def collate_fn(batch):
    input_feature_lengths = torch.tensor([i[0].shape[-1] for i in batch])
    max_len = max(input_feature_lengths)
    max_len = max_len + 32 - max_len % 32
    # padding
    for i, item in enumerate(batch):
        item = list(item)
        item[1] = torch.tensor(item[1])
        for param in [0, 2, 3]:
            item[param] = torch.nn.functional.pad(
                torch.tensor(item[param]),
                (0, max_len - item[param].shape[-1]),
                "constant",
                0,
            )
        batch[i] = tuple(item)

    ph_seq = torch.cat([item[1] for item in batch])
    ph_seq_lengths = torch.tensor([len(item[1]) for item in batch])

    input_feature = torch.stack([item[0] for item in batch])
    ph_edge = torch.stack([item[2] for item in batch])
    ph_frame = torch.stack([item[3] for item in batch])

    label_type = torch.stack([item[4] for item in batch])

    return (
        input_feature,
        input_feature_lengths,
        ph_seq,
        ph_seq_lengths,
        ph_edge,
        ph_frame,
        label_type,
    )
