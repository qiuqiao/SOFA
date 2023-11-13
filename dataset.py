import torch
import h5py
import numpy as np
import pathlib
import pandas as pd


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, binary_data_folder="data/binary", prefix="train", binning_length=1800):
        # do not open hdf5 here
        self.h5py_items = None
        self.label_types = None
        self.wav_lengths = None

        self.binary_data_folder = binary_data_folder
        self.prefix = prefix
        self.binning_length = binning_length

    def get_label_types(self):
        if self.label_types is None:
            self._open_h5py_file()
        return self.label_types

    def get_wav_lengths(self):
        if self.wav_lengths is None:
            self._open_h5py_file()
        return self.wav_lengths

    def _open_h5py_file(self):
        h5py_file = h5py.File(str(pathlib.Path(self.binary_data_folder) / (self.prefix + ".h5py")), 'r')
        self.h5py_items = h5py_file["items"]
        self.label_types = np.array(h5py_file["meta_data"]["label_types"])
        self.wav_lengths = np.array(h5py_file["meta_data"]["wav_lengths"])

    def __len__(self):
        if self.h5py_items is None:
            self._open_h5py_file()
        return len(self.h5py_items)

    def __getitem__(self, index):
        if self.h5py_items is None:
            self._open_h5py_file()

        item = self.h5py_items[str(index)]

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


class WeightedBinningAudioSampler(torch.utils.data.Sampler):
    def __init__(self, type_ids, wav_lengths, weights_of_types, max_length=100, binning_length=1000):
        assert len(weights_of_types) == max(type_ids) + 1
        assert min(type_ids) >= 0
        assert len(type_ids) == len(wav_lengths)
        assert max_length > 0
        assert binning_length > 0

        meta_data = pd.DataFrame(
            {
                "dataset_index": range(len(type_ids)),
                "type_id": type_ids,
                "wav_length": wav_lengths
            }
        )
        meta_data = meta_data.sort_values(by=["wav_length"], ascending=False).reset_index(drop=True)

        self.bins = []
        curr_bin_max_item_length = meta_data.loc[0, "wav_length"]
        curr_bin_start_index = 0
        for i in range(len(meta_data)):
            if curr_bin_max_item_length * (i - curr_bin_start_index) > binning_length:
                bin_data = meta_data.loc[curr_bin_start_index:i - 1, ].to_dict(orient="list")
                self.bins.append(bin_data)

                curr_bin_start_index = i
                curr_bin_max_item_length = meta_data.loc[i, "wav_length"]

        print(len(self.bins))


def collate_fn(batch):
    input_feature_lengths = torch.tensor([i[0].shape[-1] for i in batch])
    max_len = max(input_feature_lengths)
    max_len = max_len
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

    label_type = torch.tensor(np.array([item[4] for item in batch]))

    return (
        input_feature,
        input_feature_lengths,
        ph_seq,
        ph_seq_lengths,
        ph_edge,
        ph_frame,
        label_type,
    )


if __name__ == "__main__":
    dataset = MixedDataset()
    sampler = WeightedBinningAudioSampler(dataset.get_label_types(), dataset.get_wav_lengths(), [1, 1, 1])
