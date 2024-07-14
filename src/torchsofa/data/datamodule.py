import warnings
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm

# from torch.utils.data import DataLoader, Dataset

# class MixedDataset(Dataset):
#     def __init__(self, metadata: pd.DataFrame, vocab: dict, sample_rate: int):
#         self.metadata = metadata
#         self.vocab = vocab
#         self.sample_rate = sample_rate

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, index):
#         wav_path, ph_seq, ph_time, label_type = self.metadata.iloc[index]

#         wav, sr = torchaudio.load(wav_path)
#         ph_seq = [self.vocab[i] for i in ph_seq.split()] if label_type != 2 else None
#         ph_time = [float(i) for i in ph_time.split()] if label_type == 0 else None

#         return wav, ph_seq, ph_time, label_type


def dur_to_start_time(dur_str):
    dur = np.array([float(i) for i in dur_str.split()])
    start_time = dur.cumsum()[:-1]
    start_time = np.insert(start_time, 0, 0)
    start_time_str = " ".join([str(i) for i in start_time])
    return start_time_str


class MixedDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        preprocess: bool = True,
        data_dir: str = "data/",
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.data_path = Path(data_dir)
        self.sample_rate = sample_rate
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} not found")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"{self.data_path} is not a directory")

    def _read_transcriptions_label(self):
        print("Reading transcriptions.csv...")
        columns = ["label_type", "wav_path", "ph_seq", "ph_time"]
        # label_type: 0: full_label; 1: weak_label; 2: audio_only
        # wav_path: wav file path
        # ph_seq: phone sequence
        # ph_time: start time of each phone
        label = pd.DataFrame(columns=columns)

        trans_paths = list(self.data_path.rglob("transcriptions.csv"))
        for path in tqdm(trans_paths):
            df = pd.read_csv(path, dtype=str)
            if "name" not in df.columns:
                warnings.warn(f"{path} is not a valid transcription file")
                continue
            df.dropna(subset=["name"], inplace=True)

            # label_type
            if "ph_seq" not in df.columns:
                df["ph_seq"] = None
                df["label_type"] = 2
            elif "ph_dur" not in df.columns:
                df["label_type"] = 1
                df.loc[df["ph_seq"].isnull(), "label_type"] = 2
            else:
                df["label_type"] = 0
                df.loc[df["ph_dur"].isnull(), "label_type"] = 1
                df.loc[df["ph_seq"].isnull(), "label_type"] = 2

            # wav_path
            df["wav_path"] = df["name"].apply(
                lambda name: str(path.parent / "wavs" / name) + ".wav"
            )

            # ph_seq (does not need to convert)

            # ph_time
            df["ph_time"] = None
            if "ph_dur" in df.columns:
                df.loc[df["label_type"] == 0, "ph_time"] = df.loc[
                    df["label_type"] == 0, "ph_dur"
                ].apply(dur_to_start_time)

            label = pd.concat([label, df.loc[:, columns]], ignore_index=True)

        return label

    def prepare_data(self):
        if not self.preprocess:
            if (self.data_path / "metadata.csv").exists() and (
                self.data_path / "vocab.csv"
            ).exists():
                print("Skip preprocessing.")
                return
            else:
                warnings.warn("metadata.csv or vocab.csv not found.")

        print("Preprocessing...")

        # metadata.csv

        # read labels
        # from transcriptions.csv
        label = self._read_transcriptions_label()
        # TODO: from .TextGrid
        # TODO: from htk lab
        # TODO: combine all labels

        # process wavs and generate metadata.csv
        metadata = label
        resamples = {}
        for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
            if not Path(row["wav_path"]).exists():
                metadata.loc[i, "wav_length"] = None
                continue

            # info比load快很多，在大部分wav已经处理完毕的情况下节省大量时间
            wav_info = torchaudio.info(row["wav_path"])
            wav_length = wav_info.num_frames / self.sample_rate
            metadata.loc[i, "wav_length"] = wav_length
            if wav_info.sample_rate == self.sample_rate and wav_info.num_channels == 1:
                continue

            wav, sr = torchaudio.load(row["wav_path"])
            # mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            # resample
            if sr != self.sample_rate:
                if sr not in resamples:
                    resamples[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
                wav = resamples[sr](wav)

            torchaudio.save(row["wav_path"], wav, self.sample_rate)

        metadata.dropna(subset=["wav_length"], inplace=True)

        # TODO: train val split
        metadata["train"] = True

        # save metadata
        metadata.to_csv(self.data_path / "metadata.csv", index=False)

        # vocab.csv
        ph_set = set()
        for ph_seq in metadata["ph_seq"]:
            ph_seq = ph_seq.split()
            ph_set.update(ph_seq)
        # TODO: process ignored_phones and special_phones
        if "SP" in ph_set:
            ph_set.remove("SP")
        vocab_list = ["SP", *list(sorted(ph_set))]
        vocab_index = list(range(len(vocab_list)))
        vocab = pd.DataFrame(
            {
                "index": vocab_index,
                "phone": vocab_list,
            }
        )
        vocab.to_csv(self.data_path / "vocab.csv", index=False)

    def setup(self, stage: str):
        if stage == "fit":
            metadata = pd.read_csv(self.data_path / "metadata.csv", dtype=str)
            vocab_csv = pd.read_csv(self.data_path / "vocab.csv", dtype=str)
            vocab = {i: j for i, j in zip(vocab_csv["phone"], vocab_csv["index"])}
            vocab.update({j: i for i, j in zip(vocab_csv["phone"], vocab_csv["index"])})

            # print(metadata)
            # print(vocab)
            # self.train_set = MixedDataset(
            #     metadata[metadata["train"]], vocab, self.sample_rate
            # )

        # if stage == "test":

        # if stage == "predict":

    # def train_dataloader(self):
    #     return DataLoader(self.train_set, batch_size=32)

    # def val_dataloader(self):
    #     return DataLoader(, batch_size=32)

    # def test_dataloader(self):
    #     return DataLoader(, batch_size=32)

    # def predict_dataloader(self):
    #     return DataLoader(, batch_size=32)


if __name__ == "__main__":
    datamodule = MixedDataModule(preprocess=True, data_dir="data/")
    datamodule.prepare_data()
    datamodule.setup("fit")
