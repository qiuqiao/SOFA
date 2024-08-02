import numpy as np

from src.torchsofa.data.label import _dur_to_time, _read_from_transcriptions


def test_dur_to_time():
    ph_dur = "1 2 3"
    time_seq = _dur_to_time(ph_dur)
    assert time_seq.shape == (4,)
    assert np.all(time_seq == np.array([0.0, 1.0, 3.0, 6.0]))


def test__read_from_transcriptions():
    df = _read_from_transcriptions("data/")
    assert ["wav_path", "label_type", "ph_seq", "time_seq"] == list(df.columns)


if __name__ == "__main__":
    pass
