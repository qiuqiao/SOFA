import pathlib
import librosa
import numpy as np
import torch
import modules.rmvpe
import yaml
from tqdm import tqdm
import pandas as pd
import importlib
import h5py
import torchaudio  # TODO 发布前记得删掉

FORCED_ALIGNER_ITEM_ATTRIBUTES = [
    "input_feature",  # contentvec units or mel spectrogram or with pitch, float32[T_s, input_feature_dim]
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
    except ImportError:
        print(f"'{package_name}' not installed.")
        return False


class ForcedAlignmentBinarizer:
    def __init__(self, config: dict):
        self.config = config["global"]
        self.config["timestep"] = self.config["hop_length"] / self.config["sample_rate"]
        self.installed_torchaudio = check_and_import("torchaudio")

        self.data_folder_path = config["preprocessing"]["data_folder"]
        self.ignored_phonemes = config["preprocessing"]["ignored_phonemes"]
        self.binary_data_folder = config["global"]["binary_data_folder"]
        self.valid_set_size = config["preprocessing"]["valid_set_size"]
        self.data_folder = config["preprocessing"]["data_folder"]

    @staticmethod
    def get_vocab(data_folder_path, ignored_phonemes):
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

        # load metadata of each item
        meta_data_df = self.get_meta_data(self.data_folder)

        # split train and valid set
        valid_set_size = int(self.valid_set_size)
        meta_data_valid = (
            meta_data_df[meta_data_df["prefix"] != "no_label"]
            .sample(valid_set_size)
            .reset_index(drop=True)
        )
        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(drop=True)

        # binarize valid set
        self.binarize(
            "valid",
            meta_data_valid,
            vocab,
            self.binary_data_folder,
        )

        # binarize train set
        self.binarize(
            "train",
            meta_data_train,
            vocab,
            self.binary_data_folder,
        )

    def load_wav(self, path):
        if self.installed_torchaudio:
            waveform, sample_rate = torchaudio.load(path)
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
                path, sr=self.config["sample_rate"], mono=True
            )
            waveform = torch.from_numpy(waveform).to(self.config["device"])

        return waveform

    def get_melspec(self, waveform):
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

        return melspec_transform(waveform.unsqueeze(0)).squeeze(0)

    def binarize(
            self,
            prefix: str,
            meta_data: pd.DataFrame,
            vocab: dict,
            binary_data_folder: str,
    ):
        meta_data["ph_seq"] = meta_data["ph_seq"].apply(
            lambda x: ([vocab[i] for i in x.split(" ")] if isinstance(x, str) else [])
        )
        meta_data["ph_dur"] = meta_data["ph_dur"].apply(
            lambda x: ([float(i) for i in x.split(" ")] if isinstance(x, str) else [])
        )

        h5py_file = h5py.File(pathlib.Path(binary_data_folder) / (prefix + ".h5py"), "w")

        print("binarizing...")
        for idx, item in tqdm(meta_data.iterrows(), total=meta_data.shape[0]):
            h5py_item_data = h5py_file.create_group(str(idx))
            # input_feature: [T,input_dim]
            # melspec
            waveform = self.load_wav(item.wav_path)
            input_feature = self.get_melspec(waveform)
            T = input_feature.shape[-1]
            h5py_item_data["input_feature"] = input_feature.cpu().numpy()

            # ph_seq
            if not item.ph_seq:
                ph_seq = np.array([-1])
            else:
                ph_seq = np.array(item.ph_seq)

            h5py_item_data["ph_seq"] = ph_seq

            # ph_edge
            if not item.ph_dur:
                ph_edge = -1 * np.ones(T, dtype="int32")
            else:
                ph_edge = np.zeros(T, dtype="int32")
                ph_edge[
                    (np.array(item.ph_dur).cumsum() / self.config["timestep"]).astype(
                        "int32"
                    )[:-1]
                ] = 1

            h5py_item_data["ph_edge"] = ph_edge

            # ph_frame
            if not item.ph_dur:
                ph_frame = -1 * np.ones(T, dtype="int32")
            else:
                ph_frame = ph_seq[ph_edge.cumsum()]

            h5py_item_data["ph_frame"] = ph_frame

        h5py_file.close()

    @staticmethod
    def get_meta_data(data_folder):
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
                lambda path_: (
                    "strong_label"
                    if "strong_label" in path_
                    else "weak_label"
                    if "weak_label" in path_
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
    with open("configs/config_new.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    ForcedAlignmentBinarizer(config=cfg).process()
