import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def _dur_to_time(ph_dur: str):
    dur_seq = [float(i) for i in ph_dur.split()]
    time_seq = np.cumsum([0, *dur_seq])
    return time_seq


def _read_from_transcriptions(data_path):
    """
    读取文件夹下的transcriptions.csv，并返回合法的label
    合法的必要条件：
    1. label_type<=1，ph_seq必须存在；label_type==0，time_seq必须存在
    2. 如果label_type==0，ph_seq等于time_seq的数量减一

    columns:
        wav_path: str, wav文件路径
        label_type: int, 0: full_label; 1: weak_label; 2: audio_only
        ph_seq: np.ndarray(L,), 音素序列
        time_seq: np.ndarray(L+1,), 时间序列
    """

    print("Reading transcriptions.csv...")
    data_path = Path(data_path)
    columns = ["wav_path", "label_type", "ph_seq", "time_seq"]
    label = pd.DataFrame(columns=columns)

    trans_paths = list(data_path.rglob("transcriptions.csv"))
    for trans_path in tqdm(trans_paths):
        df = pd.read_csv(trans_path, dtype=str)
        if "name" not in df.columns:
            warnings.warn(f"{trans_path} is not a valid transcription file")
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
            lambda name: str(trans_path.parent / "wavs" / name) + ".wav"
        )

        # ph_seq
        if "ph_seq" in df.columns:
            df.loc[df["label_type"] <= 1, "ph_seq"] = df.loc[
                df["label_type"] <= 1, "ph_seq"
            ].apply(lambda ph_seq: np.array(ph_seq.split()))
        else:
            df["ph_seq"] = None

        # time_seq
        if "ph_dur" in df.columns:
            df.loc[df["label_type"] == 0, "time_seq"] = df.loc[
                df["label_type"] == 0, "ph_dur"
            ].apply(_dur_to_time)

            # len(ph_seq) == (len(time_seq) + 1)
            df["valid"] = True
            df.loc[df["label_type"] == 0, "valid"] = df.loc[
                df["label_type"] == 0, "ph_seq"
            ].apply(len) == (df.loc[df["label_type"] == 0, "time_seq"].apply(len) - 1)

            df.loc[~df["valid"], "label_type"] = 1
            df.loc[~df["valid"], "time_seq"] = None
        else:
            df["time_seq"] = None

        label = pd.concat([label, df.loc[:, columns]], ignore_index=True)

    return label


def read_labels(data_path: Union[Path, str], labels_from: List[str]):
    # TODO: from .TextGrid
    # TODO: from htk lab
    # TODO: from .wav (audio only)
    # TODO: combine all labels
    return _read_from_transcriptions(data_path)
