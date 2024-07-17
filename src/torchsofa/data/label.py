import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# def start_time_to_interval(start_time: np.ndarray, wav_length: float):
#     arr = np.append(start_time, wav_length)
#     intervals = np.stack([arr[:-1], arr[1:]]).T
#     return intervals


def str_to_floats(func):
    def wrapper(str, *args, **kwargs):
        arr = np.array([float(i) for i in str.split()])
        return func(arr, *args, **kwargs)

    return wrapper


def str_from_floats(func):
    def wrapper(str, *args, **kwargs):
        res = func(str, *args, **kwargs)
        return " ".join([f"{i:.5g}" for i in res])

    return wrapper


def dur_to_start_time(dur: np.ndarray):
    start_time = dur.cumsum()[:-1]
    start_time = np.insert(start_time, 0, 0)
    return start_time


def read_transcriptions_label(data_path):
    print("Reading transcriptions.csv...")
    columns = ["label_type", "wav_path", "ph_seq", "ph_time"]
    # label_type: 0: full_label; 1: weak_label; 2: audio_only
    # wav_path: wav file path
    # ph_seq: phone sequence
    # ph_time: start time of each phone
    label = pd.DataFrame(columns=columns)

    trans_paths = list(data_path.rglob("transcriptions.csv"))
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
            ].apply(str_from_floats(str_to_floats(dur_to_start_time)))

        label = pd.concat([label, df.loc[:, columns]], ignore_index=True)

    return label
