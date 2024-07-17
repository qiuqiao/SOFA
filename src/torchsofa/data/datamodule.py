import warnings
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm


def dur_to_start_time(dur_str):
    dur = np.array([float(i) for i in dur_str.split()])
    start_time = dur.cumsum()[:-1]
    start_time = np.insert(start_time, 0, 0)
    start_time_str = " ".join([f"{i:.5g}" for i in start_time])
    return start_time_str


class MixedDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        preprocess: bool = True,
        data_dir: str = "data/",
        sample_rate: int = 16000,
        max_length: float = 50.0,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.data_path = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
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
            if (self.data_path / "metadata.csv").exists():
                print("Skip preprocessing.")
                return
            else:
                warnings.warn("metadata.csv not found.")

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
            if wav_length > self.max_length:
                warnings.warn(f"{row['wav_path']} is too long. Skip.")
                continue
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

        # save metadata
        metadata = metadata.sort_values(by="wav_length", ascending=False)
        metadata.to_csv(self.data_path / "metadata.csv", index=False)

        # save statistic data
        total_length = metadata["wav_length"].sum()

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.hist(
            metadata["wav_length"], bins=50, alpha=0.7, color="blue", edgecolor="black"
        )
        ax.set_xlabel("Length (seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Distribution of Wav Lengths, total: {total_length/3600:.2f} hours"
        )
        plt.savefig(self.data_path / "wav_length_distribution.png")
        print(f"Total length of wavs: {total_length/3600:.2f} hours.")
        print(
            f"Saved wav length distribution to {self.data_path / 'wav_length_distribution.png'}."
        )

    def setup(self, stage: str):
        if stage == "fit":
            metadata = pd.read_csv(self.data_path / "metadata.csv", dtype=str)

            # print(metadata)

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
