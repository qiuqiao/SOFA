import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm


def dur_to_start_time(dur_str):
    dur = np.array([float(i) for i in dur_str.split()])
    start_time = dur.cumsum()[:-1]
    start_time = np.insert(start_time, 0, 0)
    start_time_str = " ".join([f"{i:.5g}" for i in start_time])
    return start_time_str


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
            ].apply(dur_to_start_time)

        label = pd.concat([label, df.loc[:, columns]], ignore_index=True)

    return label
