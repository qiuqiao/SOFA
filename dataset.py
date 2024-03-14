import pathlib

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange


class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        augmentation_size,
        binary_data_folder="data/binary",
        prefix="train",
    ):
        # do not open hdf5 here
        self.h5py_file = None
        self.label_types = None
        self.wav_lengths = None
        if augmentation_size > 0:
            self.augmentation_indexes = np.arange(augmentation_size + 1)
        else:
            self.augmentation_indexes = None

        self.binary_data_folder = binary_data_folder
        self.prefix = prefix

    def get_label_types(self):
        uninitialized = self.label_types is None
        if uninitialized:
            self._open_h5py_file()
        ret = self.label_types
        if uninitialized:
            self._close_h5py_file()
        return ret

    def get_wav_lengths(self):
        uninitialized = self.wav_lengths is None
        if uninitialized:
            self._open_h5py_file()
        ret = self.wav_lengths
        if uninitialized:
            self._close_h5py_file()
        return ret

    def _open_h5py_file(self):
        self.h5py_file = h5py.File(
            str(pathlib.Path(self.binary_data_folder) / (self.prefix + ".h5py")), "r"
        )
        self.label_types = np.array(self.h5py_file["meta_data"]["label_types"])
        self.wav_lengths = np.array(self.h5py_file["meta_data"]["wav_lengths"])

    def _close_h5py_file(self):
        self.h5py_file.close()
        self.h5py_file = None

    def __len__(self):
        uninitialized = self.h5py_file is None
        if uninitialized:
            self._open_h5py_file()
        ret = len(self.h5py_file["items"])
        if uninitialized:
            self._close_h5py_file()
        return ret

    def __getitem__(self, index):
        if self.h5py_file is None:
            self._open_h5py_file()

        item = self.h5py_file["items"][str(index)]

        # input_feature
        if self.augmentation_indexes is None:
            input_feature = np.array(item["input_feature"])
        else:
            indexes = np.random.choice(self.augmentation_indexes, 2)
            input_feature = np.array(item["input_feature"])[indexes, :, :]

        # label_type
        label_type = np.array(item["label_type"])

        # ph_seq
        ph_seq = np.array(item["ph_seq"])

        # ph_edge
        ph_edge = np.array(item["ph_edge"])

        # ph_frame
        ph_frame = np.array(item["ph_frame"])

        # ph_mask
        ph_mask = np.array(item["ph_mask"])

        input_feature = np.repeat(
            input_feature, len(ph_frame) // input_feature.shape[-1], axis=-1
        )

        return input_feature, ph_seq, ph_edge, ph_frame, ph_mask, label_type


class WeightedBinningAudioBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        type_ids,
        wav_lengths,
        oversampling_weights=None,
        max_length=100,
        binning_length=1000,
        drop_last=False,
    ):
        if oversampling_weights is None:
            oversampling_weights = [1] * (max(type_ids) + 1)
        oversampling_weights = np.array(oversampling_weights).astype(np.float32)

        assert min(oversampling_weights) > 0
        assert len(oversampling_weights) >= max(type_ids) + 1
        assert min(type_ids) >= 0
        assert len(type_ids) == len(wav_lengths)
        assert max_length > 0
        assert binning_length > 0

        count = np.bincount(type_ids)
        count = np.pad(count, (0, len(oversampling_weights) - len(count)))
        self.oversampling_weights = oversampling_weights / min(
            oversampling_weights[count > 0]
        )
        self.max_length = max_length
        self.drop_last = drop_last

        # sort by wav_lengths
        meta_data = (
            pd.DataFrame(
                {
                    "dataset_index": range(len(type_ids)),
                    "type_id": type_ids,
                    "wav_length": wav_lengths,
                }
            )
            .sort_values(by=["wav_length"], ascending=False)
            .reset_index(drop=True)
        )

        # binning and compute oversampling num
        self.bins = []

        curr_bin_start_index = 0
        curr_bin_max_item_length = meta_data.loc[0, "wav_length"]
        for i in range(len(meta_data)):
            if curr_bin_max_item_length * (i - curr_bin_start_index) > binning_length:
                bin_data = {
                    "batch_size": self.max_length // curr_bin_max_item_length,
                    "num_batches": 0,
                    "type": [],
                }

                item_num = 0
                for type_id, weight in enumerate(self.oversampling_weights):
                    idx_list = (
                        meta_data.loc[curr_bin_start_index : i - 1]
                        .loc[meta_data["type_id"] == type_id]
                        .to_dict(orient="list")["dataset_index"]
                    )

                    oversample_num = np.round(len(idx_list) * (weight - 1))
                    bin_data["type"].append(
                        {
                            "idx_list": idx_list,
                            "oversample_num": oversample_num,
                        }
                    )
                    item_num += len(idx_list) + oversample_num

                if bin_data["batch_size"] <= 0:
                    raise ValueError(
                        "batch_size <= 0, maybe batch_max_length in training config is too small "
                        "or max_length in binarizing config is too long."
                    )
                num_batches = item_num / bin_data["batch_size"]
                if self.drop_last:
                    bin_data["num_batches"] = int(num_batches)
                else:
                    bin_data["num_batches"] = int(np.ceil(num_batches))
                self.bins.append(bin_data)

                curr_bin_start_index = i
                curr_bin_max_item_length = meta_data.loc[i, "wav_length"]

        self.len = None

    def __len__(self):
        if self.len is None:
            self.len = 0
            for bin_data in self.bins:
                self.len += bin_data["num_batches"]
        return self.len

    def __iter__(self):
        np.random.shuffle(self.bins)

        for bin_data in self.bins:
            batch_size = bin_data["batch_size"]
            num_batches = bin_data["num_batches"]

            idx_list = []
            for type_id, weight in enumerate(self.oversampling_weights):
                idx_list_of_type = bin_data["type"][type_id]["idx_list"]
                oversample_num = bin_data["type"][type_id]["oversample_num"]

                if len(idx_list_of_type) > 0:
                    idx_list.extend(idx_list_of_type)
                    oversample_idx_list = np.random.choice(
                        idx_list_of_type, int(oversample_num)
                    )
                    idx_list.extend(oversample_idx_list)

            idx_list = np.random.permutation(idx_list)

            if self.drop_last:
                num_batches = int(num_batches)
                idx_list = idx_list[: num_batches * batch_size]
            else:
                num_batches = int(np.ceil(num_batches))
                random_idx = np.random.choice(
                    idx_list, int(num_batches * batch_size - len(idx_list))
                )
                idx_list = np.concatenate([idx_list, random_idx])

            np.random.shuffle(idx_list)

            for i in range(num_batches):
                yield idx_list[int(i * batch_size) : int((i + 1) * batch_size)]


def collate_fn(batch):
    """_summary_

    Args:
        batch (tuple): input_feature, ph_seq, ph_edge, ph_frame, ph_mask, label_type from MixedDataset

    Returns:
        input_feature: (B C T)
        input_feature_lengths: (B)
        ph_seq: (B S)
        ph_seq_lengths: (B)
        ph_edge: (B T)
        ph_frame: (B T)
        ph_mask: (B vocab_size)
        label_type: (B)
    """
    input_feature_lengths = torch.tensor([i[0].shape[-1] for i in batch])
    max_len = max(input_feature_lengths)
    ph_seq_lengths = torch.tensor([len(item[1]) for item in batch])
    max_ph_seq_len = max(ph_seq_lengths)
    if batch[0][0].shape[0] > 1:
        augmentation_enabled = True
    else:
        augmentation_enabled = False

    # padding
    for i, item in enumerate(batch):
        item = list(item)
        for param in [0, 2, 3]:
            item[param] = torch.nn.functional.pad(
                torch.tensor(item[param]),
                (0, max_len - item[param].shape[-1]),
                "constant",
                0,
            )
        item[1] = torch.nn.functional.pad(
            torch.tensor(item[1]),
            (0, max_ph_seq_len - item[1].shape[-1]),
            "constant",
            0,
        )
        item[4] = torch.from_numpy(item[4])
        batch[i] = tuple(item)

    input_feature = torch.stack([item[0] for item in batch], dim=1)
    input_feature = rearrange(input_feature, "n b c t -> (n b) c t")
    ph_seq = torch.stack([item[1] for item in batch])
    ph_edge = torch.stack([item[2] for item in batch])
    ph_frame = torch.stack([item[3] for item in batch])
    ph_mask = torch.stack([item[4] for item in batch])

    label_type = torch.tensor(np.array([item[5] for item in batch]))

    if augmentation_enabled:
        input_feature_lengths = torch.concat(
            [input_feature_lengths, input_feature_lengths], dim=0
        )
        ph_seq = torch.concat([ph_seq, ph_seq], dim=0)
        ph_seq_lengths = torch.concat([ph_seq_lengths, ph_seq_lengths], dim=0)
        ph_edge = torch.concat([ph_edge, ph_edge], dim=0)
        ph_frame = torch.concat([ph_frame, ph_frame], dim=0)
        ph_mask = torch.concat([ph_mask, ph_mask], dim=0)
        label_type = torch.concat([label_type, label_type], dim=0)

    return (
        input_feature,
        input_feature_lengths,
        ph_seq,
        ph_seq_lengths,
        ph_edge,
        ph_frame,
        ph_mask,
        label_type,
    )


if __name__ == "__main__":
    dataset = MixedDataset(2)
    print(dataset[0])
    # sampler = WeightedBinningAudioBatchSampler(dataset.get_label_types(), dataset.get_wav_lengths(), [1, 0.3, 0.4])
    # for i in tqdm(sampler):
    #     print(len(i))
