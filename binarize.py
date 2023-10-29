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
import click
import torchaudio

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
        print("Generating vocab...")
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
            meta_data_df[meta_data_df["label_type"] != "no_label"]
            .sample(valid_set_size)
        )
        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(drop=True)
        meta_data_valid = meta_data_valid.reset_index(drop=True)

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
        print(f"Binarizing {prefix} set...")
        meta_data["ph_seq"] = meta_data["ph_seq"].apply(
            lambda x: ([vocab[i] for i in x.split(" ")] if isinstance(x, str) else [])
        )
        meta_data["ph_dur"] = meta_data["ph_dur"].apply(
            lambda x: ([float(i) for i in x.split(" ")] if isinstance(x, str) else [])
        )
        meta_data = meta_data.sort_values(by="label_type").reset_index(drop=True)

        h5py_file_path = pathlib.Path(binary_data_folder) / (prefix + ".h5py")
        h5py_file = h5py.File(h5py_file_path, "w")
        h5py_items = h5py_file.create_group("items")

        label_type_to_id = {"no_label": 0, "weak_label": 1, "strong_label": 2}
        label_type_ids = []

        idx = 0
        total_timestep = 0
        for _, item in tqdm(meta_data.iterrows(), total=meta_data.shape[0]):

            # input_feature: [input_dim,T]
            waveform = self.load_wav(item.wav_path)
            input_feature = self.get_melspec(waveform)

            T = input_feature.shape[-1]
            if T > self.config["max_timestep"]:
                print(f"Item {item.path} has a length of{T * self.config['max_timestep']} is too long, skip it.")
                continue

            else:
                h5py_item_data = h5py_items.create_group(str(idx))
                idx += 1
                total_timestep += T

            h5py_item_data["input_feature"] = input_feature.cpu().numpy().astype("float32")

            # label_type: []
            label_type_id = label_type_to_id[item.label_type]
            h5py_item_data["label_type"] = label_type_id
            label_type_ids.append(label_type_id)

            # ph_seq: [S]
            if label_type_id < 1:
                ph_seq = np.array([])
            else:
                ph_seq = np.array(item.ph_seq)

            h5py_item_data["ph_seq"] = ph_seq.astype("int32")

            # ph_edge: [T]
            if label_type_id < 2:
                ph_edge = np.zeros(T, dtype="float32")
            else:
                ph_edge = np.zeros(T, dtype="float32")
                ph_edge_int = np.zeros(T, dtype="int32")  # for ph_frame
                ph_time = (np.array(item.ph_dur).cumsum() / self.config["timestep"])[:-1]
                if ph_time[0] < 0.5 + 1e-3:
                    ph_time = ph_time[1:]
                ph_time_int = ph_time.round().astype("int32")
                ph_time_fractional = ph_time - ph_time_int
                ph_edge_int[ph_time_int] = 1
                ph_edge[ph_time_int] = 0.5 + ph_time_fractional
                ph_edge[ph_time_int - 1] = 0.5 - ph_time_fractional

            h5py_item_data["ph_edge"] = ph_edge.astype("float32")

            # ph_frame: [T]
            if label_type_id < 2:
                ph_frame = np.zeros(T, dtype="int32")
            else:
                ph_frame = ph_seq[ph_edge_int.cumsum().astype("int32")]

            h5py_item_data["ph_frame"] = ph_frame.astype("int32")

            # print(h5py_item_data["input_feature"].shape,
            #       h5py_item_data["label_type"].shape,
            #       h5py_item_data["ph_seq"].shape,
            #       h5py_item_data["ph_edge"].shape,
            #       h5py_item_data["ph_frame"].shape
            #       )

        h5py_file.create_dataset("label_type_ids", data=np.array(label_type_ids).astype("int32"))
        h5py_file.close()
        total_time = total_timestep * self.config["timestep"]
        print(
            f"Successfully binarized {prefix} set, total time {total_time:.2f}s, saved to {h5py_file_path}"
        )

    @staticmethod
    def get_meta_data(data_folder):
        path = pathlib.Path(data_folder)
        trans_path_list = [
            i
            for i in path.rglob("transcriptions.csv")
            if i.name == "transcriptions.csv"
        ]

        print("Loading metadata...")
        meta_data_df = pd.DataFrame()
        for trans_path in tqdm(trans_path_list):
            df = pd.read_csv(trans_path)
            df["wav_path"] = df["name"].apply(
                lambda name: str(trans_path.parent / "wavs" / (str(name) + ".wav")),
            )
            df["label_type"] = df["wav_path"].apply(
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
        meta_data_df["label_type"].fillna("no_label", inplace=True)

        meta_data_df.reset_index(drop=True, inplace=True)

        return meta_data_df


@click.command()
@click.option("--config_path", "-c", type=str, default="configs/config.yaml", show_default=True, help="config path")
def binarize(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # print(cfg)
    ForcedAlignmentBinarizer(config=cfg).process()


if __name__ == "__main__":
    binarize()
