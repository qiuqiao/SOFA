import numpy as np
import pandas as pd

from src.torchsofa.data.manager import DataManager


class TestDataManager:
    def test_serialize(self):
        data = DataManager()
        data.df = pd.DataFrame(
            {
                "wav_path": ["test.wav"],
                "label_type": [0],
                "ph_seq": [np.array(["a", "b", "c"])],
                "time_seq": [np.array([0.0, 1.111111111, 3.999999999])],
                "wav_length": [3.0],
            }
        )
        data.serialize("test.csv")

        data_ = DataManager().deserialize("test.csv")
        data_.serialize("test_.csv")

        with open("test.csv", "r") as f:
            with open("test_.csv", "r") as f_:
                assert f.read() == f_.read()

    def test_save_statistics(self):
        data = DataManager()
        data.df = pd.DataFrame(
            {
                "wav_path": ["test.wav"],
                "label_type": [0],
                "ph_seq": [np.array(["a", "b", "c"])],
                "time_seq": [np.array([0.0, 1.111111111, 3.999999999])],
                "wav_length": [3.0],
            }
        )

        data.save_statistics("test.png")

    def test_get_phone_set(self):
        data = DataManager()
        data.df = pd.DataFrame(
            {
                "wav_path": ["test.wav", "test_2.wav"],
                "label_type": [0, 1],
                "ph_seq": [np.array(["a", "b", "c"]), np.array(["a", "d", "e"])],
                "time_seq": [
                    np.array([0.0, 1.111111111, 3.999999999, 5]),
                    np.array([0.0, 1.0, 3.0, 4]),
                ],
                "wav_length": [5.0, 4.0],
            }
        )

        assert data.get_phone_set() == {"a", "b", "c", "d", "e"}


if __name__ == "__main__":
    TestDataManager().test_get_phone_set()
