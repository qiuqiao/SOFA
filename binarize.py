import copy
import csv
import json
import pathlib
import random
import librosa
import numpy as np
import torch
from scipy import interpolate
import modules.rmvpe
import yaml
from tqdm import tqdm
import pandas as pd
import importlib
import torchaudio  # *发布前记得删掉
import parselmouth

FORCED_ALIGNER_ITEM_ATTRIBUTES = [
    "input_feature",  # contentvec units or mel spectroogram and putch, float32[T_s, input_feature_dim]
    "ph_seq",  # phoneme sequence, str[T_p,]
    "ph_edge",  # edge of phoneme, int32[T_s,], if the label does not exist, it is all -1
    "ph_frame",  # frame-wise phoneme class, int32[T_s,], if the label does not exist, it is all -1
]

melspec_transform = None
resample_transform_dict = {}
rmvpe = None


def check_and_import(package_name):
    try:
        importlib.import_module(package_name)
        globals()[package_name] = importlib.import_module(package_name)
        print(f"'{package_name}' installed and imported.")
        return True
    except:
        print(f"'{package_name}' not installed.")
        return False


class ForcedAlignmentBinarizer:
    def __init__(self, config: dict):
        self.config = config["global"]
        self.config["timestep"] = self.config["hop_length"] / self.config["sample_rate"]
        self.installed_torchaudio = check_and_import("torchaudio")

        self.data_folder_path = config["preprocessing"]["data_folder"]
        self.ignored_phonemes = config["preprocessing"]["ignored_phonemes"]
        self.binary_data_folder = config["preprocessing"]["binary_data_folder"]
        self.valid_set_size = config["preprocessing"]["valid_set_size"]
        self.data_folder = config["preprocessing"]["data_folder"]
        self.pitch_extractor = config["preprocessing"]["pitch_extractor"]

    def get_vocab(self, data_folder_path, ignored_phonemes):
        print("generating vocab...")
        phonemes = []
        trans_path_list = pathlib.Path(data_folder_path).rglob("transcriptions.csv")

        for trans_path in trans_path_list:
            if trans_path.name == "transcriptions.csv":
                df = pd.read_csv(trans_path)
                ph = list(set(" ".join(df["ph_seq"]).split(" ")))
                phonemes.extend(ph)

        phonemes = set(phonemes)
        for p in ignored_phonemes:
            if p in phonemes:
                phonemes.remove(p)
        phonemes = sorted(phonemes)
        phonemes = ["<EMPTY>", *phonemes]

        vocab = dict(zip(phonemes, range(len(phonemes))))
        vocab.update(dict(zip(range(len(phonemes)), phonemes)))
        vocab.update({i: 0 for i in ignored_phonemes})
        vocab.update({"<vocab_size>": len(phonemes)})

        print(f"vocab_size is {len(phonemes)}")

        return vocab

    def process(self):
        vocab = self.get_vocab(self.data_folder_path, self.ignored_phonemes)
        with open(pathlib.Path(self.binary_data_folder) / "vocab.yaml", "w") as file:
            yaml.dump(vocab, file)

        # load meta data of each item
        meta_data_df = self.get_meta_data(self.data_folder)

        # split train and valid set
        valid_set_size = int(self.valid_set_size)
        meta_data_valid = meta_data_df[meta_data_df["prefix"] != "no_label"].sample(
            valid_set_size
        )
        meta_data_train = meta_data_df.drop(meta_data_valid.index)

        # binarize valid set
        self.binarize(
            "valid",
            meta_data_valid,
            vocab,
            self.binary_data_folder,
            self.pitch_extractor,
        )

        # binarize train set
        self.binarize(
            "train",
            meta_data_train,
            vocab,
            self.binary_data_folder,
            self.pitch_extractor,
        )

    def binarize(
        self,
        prefix: str,
        meta_data: pd.DataFrame,
        vocab: dict,
        binary_data_folder: str,
        pitch_extractor,
    ):
        meta_data["ph_seq"] = meta_data["ph_seq"].apply(
            lambda x: ([vocab[i] for i in x.split(" ")] if isinstance(x, str) else [])
        )
        meta_data["ph_dur"] = meta_data["ph_dur"].apply(
            lambda x: ([float(i) for i in x.split(" ")] if isinstance(x, str) else [])
        )

        idx_data = []
        binary_file = open(pathlib.Path(binary_data_folder) / (prefix + ".data"), "wb")

        print("binarizing...")
        for idx, item in tqdm(meta_data.iterrows(), total=meta_data.shape[0]):
            idx_data_item = {}

            # input_feature: [input_dim,T]
            # melspec
            if self.installed_torchaudio:
                waveform, sample_rate = torchaudio.load(item.wav_path)
                if self.config["sample_rate"] != sample_rate:
                    global resample_transform_dict
                    if sample_rate not in resample_transform_dict:
                        resample_transform_dict[
                            sample_rate
                        ] = torchaudio.transforms.Resample(
                            sample_rate, self.config["sample_rate"]
                        )

                    waveform = resample_transform_dict[sample_rate](waveform)

                waveform = waveform[0].to(self.config["device"])

            else:
                waveform, _ = librosa.load(
                    item.wav_path, sr=self.config["sample_rate"], mono=True
                )
                waveform = torch.from_numpy(waveform).to(self.config["device"])

            global melspec_transform
            if melspec_transform is None:
                melspec_transform = modules.rmvpe.MelSpectrogram(
                    n_mel_channels=self.config["n_mels"],
                    sampling_rate=self.config["sample_rate"],
                    win_length=self.config["win_length"],
                    hop_length=self.config["hop_length"],
                    mel_fmin=self.config["fmin"],
                    mel_fmax=self.config["fmax"],
                ).to(self.config["device"])
            input_feature = (
                melspec_transform(waveform.unsqueeze(0)).squeeze(0).cpu().numpy()
            )
            T = input_feature.shape[-1]
            # *pitch

            idx_data_item["input_feature"] = {}
            self.wirte_ndarray_to_bin(
                binary_file, idx_data_item["input_feature"], input_feature
            )

            # ph_seq
            if item.ph_seq == []:
                ph_seq = np.array([-1])
            else:
                ph_seq = np.array(item.ph_seq)

            idx_data_item["ph_seq"] = {}
            self.wirte_ndarray_to_bin(binary_file, idx_data_item["ph_seq"], ph_seq)

            # ph_edge
            if item.ph_dur == []:
                ph_edge = -1 * np.ones(T, dtype="int32")
            else:
                ph_edge = np.zeros(T, dtype="int32")
                ph_edge[
                    (np.array(item.ph_dur).cumsum() / self.config["timestep"]).astype(
                        "int32"
                    )[:-1]
                ] = 1

            idx_data_item["ph_edge"] = {}
            self.wirte_ndarray_to_bin(binary_file, idx_data_item["ph_edge"], ph_edge)

            # ph_frame
            if item.ph_dur == []:
                ph_frame = -1 * np.ones(T, dtype="int32")
            else:
                ph_frame = ph_seq[ph_edge.cumsum()]

            idx_data_item["ph_frame"] = {}
            self.wirte_ndarray_to_bin(binary_file, idx_data_item["ph_frame"], ph_frame)

            idx_data.append(idx_data_item)

        idx_data = pd.DataFrame(idx_data)
        idx_data.to_pickle(pathlib.Path(binary_data_folder) / (prefix + ".idx"))
        binary_file.close()

    def wirte_ndarray_to_bin(self, file, idx_data, array):
        idx_data["start"] = file.tell()
        idx_data["shape"] = array.shape
        idx_data["dtype"] = str(array.dtype)

        array = array.tobytes()
        file.write(array)
        idx_data["len"] = file.tell() - idx_data["start"]

    def get_meta_data(self, data_folder):
        path = pathlib.Path(data_folder)
        trans_path_list = [
            i
            for i in path.rglob("transcriptions.csv")
            if i.name == "transcriptions.csv"
        ]

        print("loading metadata...")
        meta_data_df = pd.DataFrame()
        for trans_path in tqdm(trans_path_list):
            df = pd.read_csv(trans_path)
            df["wav_path"] = df["name"].apply(
                lambda name: str(trans_path.parent / "wavs" / (str(name) + ".wav")),
            )
            df["prefix"] = df["wav_path"].apply(
                lambda path: (
                    "strong_label"
                    if "strong_label" in path
                    else "weak_label"
                    if "weak_label" in path
                    else "no_label"
                ),
            )
            meta_data_df = pd.concat([meta_data_df, df])

        no_label_df = pd.DataFrame(
            {"wav_path": [i for i in (path / "no_label").rglob("*.wav")]}
        )
        meta_data_df = pd.concat([meta_data_df, no_label_df])
        meta_data_df["prefix"].fillna("no_label", inplace=True)

        meta_data_df.reset_index(drop=True, inplace=True)

        return meta_data_df


if __name__ == "__main__":
    with open("configs/config_new.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # print(config)
    ForcedAlignmentBinarizer(config=config).process()
