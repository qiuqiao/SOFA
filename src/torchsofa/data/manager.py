import time
import warnings
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm

from .label import read_labels
from .vocab import Vocab


class DataManager:
    def __init__(self):
        self.df = None

    def get_dataframe(self):
        if self.df is not None:
            return self.df
        raise ValueError("Data not prepared or deserialized from csv file.")

    def prepare(
        self,
        data_path: Union[Path, str],
        sample_rate: int,
        max_length: float,
        labels_from: list = ["transcriptions", "textgrid", "htk"],
    ):
        data_path = Path(data_path)

        # 扔掉labels中文件不存在的wav
        wav_paths = list(data_path.rglob("*.wav"))
        labels = read_labels(data_path, labels_from)
        wav_paths_set = set([str(i) for i in wav_paths])
        wav_paths_set_from_label = set(labels["wav_path"])
        intersection = wav_paths_set & wav_paths_set_from_label
        labels = labels[labels["wav_path"].isin(intersection)]

        # 加入labels中没有，但存在文件的wav
        difference = wav_paths_set - wav_paths_set_from_label
        df = pd.DataFrame(
            {
                "wav_path": [str(i) for i in difference],
                "label_type": [2] * len(difference),
            }
        )
        labels = pd.concat([labels, df], ignore_index=True)

        # 统一wav格式，并获取wav_length，同时扔掉torchaudio读取出错的wav和不符合要求的wav
        wav_lengths = []

        resamples = {}
        wav_paths = labels["wav_path"].tolist()
        print("Preprocessing wavs...")
        for wav_path in tqdm(wav_paths):
            # wav_length
            try:
                wav_info = torchaudio.info(wav_path)

                wav_length = wav_info.num_frames / wav_info.sample_rate
                if wav_length > max_length:
                    warnings.warn(
                        f"{wav_path} has a duration of {wav_length}s, longer than {max_length}s. Skip."
                    )
                    wav_lengths.append(None)  # 表示wav不符合要求，丢弃
                    continue
                wav_lengths.append(wav_length)
            except Exception as e:
                warnings.warn(f"{wav_path} is invalid. Skip. Error message: {e}.")
                wav_lengths.append(None)
                continue

            try:
                if wav_info.sample_rate == sample_rate and wav_info.num_channels == 1:
                    continue
                wav, sr = torchaudio.load(wav_path)
                # mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                # sample rate
                if sr != sample_rate:
                    if sr not in resamples:
                        resamples[sr] = torchaudio.transforms.Resample(sr, sample_rate)
                    wav = resamples[sr](wav)
                torchaudio.save(wav_path, wav, sample_rate)

            except Exception as e:
                warnings.warn(f"{wav_path} is invalid. Skip. Error message: {e}.")
                wav_lengths[-1] = None

        labels["wav_length"] = wav_lengths
        labels = labels[labels["wav_length"].notnull()]

        # 根据文件夹判断label_type，只能降级不能升级
        def get_label_type_from_path(wav_path):
            path_parts = Path(wav_path).parts[:-1]
            for path_part in path_parts[::-1]:
                if path_part == "full_label":
                    return 0
                if path_part == "weak_label":
                    return 1
                if path_part == "audio_only" or path_part == "no_label":
                    return 2
            return 2

        label_type_from_path = labels["wav_path"].map(get_label_type_from_path)
        is_bigger = label_type_from_path > labels["label_type"]
        labels.loc[is_bigger, "label_type"] = label_type_from_path[is_bigger]

        # columns: wav_path label_type ph_seq time_seq wav_length
        self.df = labels.reset_index(drop=True)

        return self

    def serialize(self, path: Union[Path, str]):
        columns = ["wav_path", "label_type", "ph_seq", "time_seq", "wav_length"]
        df = self.df[columns]

        df["wav_path"].apply(lambda wav_path: str(wav_path))

        df.loc[df["label_type"] <= 1, "ph_seq"] = df.loc[
            df["label_type"] <= 1, "ph_seq"
        ].apply(lambda ph_seq: " ".join(ph_seq))

        df.loc[df["label_type"] == 0, "time_seq"] = df.loc[
            df["label_type"] == 0, "time_seq"
        ].apply(lambda time_seq: " ".join([f"{i:.4g}" for i in time_seq]))

        df["wav_length"] = df["wav_length"].apply(lambda wav_length: f"{wav_length:4g}")

        df.to_csv(path, index=False)

    def deserialize(self, path: Union[Path, str]):
        df = pd.read_csv(
            path,
            dtype={
                "wav_path": str,
                "label_type": int,
                "ph_seq": str,
                "time_seq": str,
                "wav_length": float,
            },
        )

        df["wav_path"] = df["wav_path"].apply(lambda wav_path: Path(wav_path))

        df.loc[df["label_type"] <= 1, "ph_seq"] = df.loc[
            df["label_type"] <= 1, "ph_seq"
        ].apply(lambda ph_seq: np.array(ph_seq.split()))

        df.loc[df["label_type"] == 0, "time_seq"] = df.loc[
            df["label_type"] == 0, "time_seq"
        ].apply(lambda time_seq: np.array([float(i) for i in time_seq.split()]))

        self.df = df

        return self

    def save_statistics(self, png_path: Union[Path, str]):
        total_length = self.df["wav_length"].sum()

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.hist(
            self.df["wav_length"], bins=50, alpha=0.7, color="blue", edgecolor="black"
        )
        ax.set_xlabel("Length (seconds)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"Distribution of Wav Lengths, total: {total_length/3600:.2f} hours"
        )
        plt.savefig(png_path)
        print(f"Total length of wavs: {total_length/3600:.2f} hours.")
        print(f"Saved wav length distribution to {png_path}.")

    def get_phone_set(self):
        phones = set()
        for ph_seq in self.df["ph_seq"].dropna():
            phones.update(ph_seq)
        return phones

    def apply_vocab(self, vocab: Vocab):
        # 得到 normal_id_seq normal_interval_seq special_id_seq special_interval_seq
        # 转换，phone转换为id，time转成interval
        start_time = time.time()

        normal_id_seqs = []
        normal_interval_seqs = []
        special_id_seqs = []
        special_interval_seqs = []
        for _, row in tqdm(
            self.df.loc[:, ["label_type", "ph_seq", "time_seq"]].iterrows(),
            total=len(self.df),
        ):
            label_type = row["label_type"]
            if label_type >= 2:
                normal_id_seqs.append(None)
                normal_interval_seqs.append(None)
                special_id_seqs.append(None)
                special_interval_seqs.append(None)
                continue
            id_seq = np.array(vocab.get_ids(row["ph_seq"]))
            # 把ignored扔掉
            not_ignored = id_seq != -1
            id_seq = id_seq[not_ignored]
            # 再把special分离出来
            is_special = np.array(vocab.is_specials(row["ph_seq"]))
            special_id_seq = id_seq[is_special]
            normal_id_seq = id_seq[~is_special]

            normal_id_seqs.append(normal_id_seq.tolist())
            special_id_seqs.append(special_id_seq.tolist())

            if label_type == 0:
                interval_seq = np.stack(
                    (row["time_seq"][:-1], row["time_seq"][1:]), axis=1
                )
                interval_seq = interval_seq[not_ignored]
                special_interval_seq = interval_seq[is_special]
                normal_interval_seq = interval_seq[~is_special]

                normal_interval_seqs.append(normal_interval_seq.tolist())
                special_interval_seqs.append(special_interval_seq.tolist())
            else:
                normal_interval_seqs.append(None)
                special_interval_seqs.append(None)

        df = pd.DataFrame(
            {
                "normal_id_seq": normal_id_seqs,
                "normal_interval_seq": normal_interval_seqs,
                "special_id_seq": special_id_seqs,
                "special_interval_seq": special_interval_seqs,
            }
        )
        df.index = self.df.index
        self.df = pd.concat([self.df, df], axis=1)

        run_time = time.time() - start_time
        print(f"Apply vocab to weak label took {run_time:.2f} seconds.")

    def apply_train_val_split(
        self,
        size: int = 15,
        preferred: List[str] = ["valid", "test"],
        random_seed: int = 114514,
    ):
        # 匹配文件名和所有父文件夹，层级越低优先级越高
        # 仅在full_label和weak_label中选择
        def match(path_parts, preferred):
            for i, path_part in enumerate(path_parts[::-1]):
                if path_part in preferred:
                    return i
            return len(path_parts)

        path_parts_series = self.df.loc[self.df["label_type"] <= 1, "wav_path"].map(
            lambda path: path.parts
        )
        priority_series = path_parts_series.map(
            lambda path_parts: match(path_parts, preferred)
        )

        self.df.loc[self.df["label_type"] <= 1, "valid_priority"] = priority_series
        # 按照优先级排序
        self.df = self.df.sample(frac=1, random_state=random_seed)
        self.df.sort_values(by="valid_priority", inplace=True, ascending=True)
        # 划分数据集
        self.df["is_valid"] = False
        self.df.iloc[:size, -1] = True

        self.df.drop(columns=["valid_priority"], inplace=True)

    def get_train_set(self):
        raise NotImplementedError

    def get_val_set(self):
        raise NotImplementedError
