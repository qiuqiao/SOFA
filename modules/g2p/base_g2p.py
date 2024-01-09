import pandas as pd
import torch


class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataset = dataframe

    def __getitem__(self, index):
        return tuple(self.dataset.iloc[index])

    def __len__(self):
        return len(self.dataset)


class BaseG2P:
    def __init__(self, **kwargs):
        # args: list of str
        self.in_format = "lab"

    def _g2p(self, input_text):
        # input text, return phoneme sequence, word sequence, and phoneme index to word index mapping
        # ph_seq: list of phonemes, SP is the silence phoneme.
        # word_seq: list of words.
        # ph_idx_to_word_idx: ph_idx_to_word_idx[i] = j means the i-th phoneme belongs to the j-th word.
        # if ph_idx_to_word_idx[i] = -1, the i-th phoneme is a silence phoneme.
        # example: ph_seq = ['SP', 'ay', 'SP', 'ae', 'm', 'SP', 'ah', 'SP', 's', 't', 'uw', 'd', 'ah', 'n', 't', 'SP']
        #          word_seq = ['I', 'am', 'a', 'student']
        #          ph_idx_to_word_idx = [-1, 0, -1, 1, 1, -1, 2, -1, 3, 3, 3, 3, 3, 3, 3, -1]
        raise NotImplementedError

    def __call__(self, text):
        ph_seq, word_seq, ph_idx_to_word_idx = self._g2p(text)

        # The first and last phonemes should be `SP`,
        # and there should not be more than two consecutive `SP`s at any position.
        assert ph_seq[0] == "SP" and ph_seq[-1] == "SP"
        assert all(
            ph_seq[i] != "SP" or ph_seq[i + 1] != "SP" for i in range(len(ph_seq) - 1)
        )
        return ph_seq, word_seq, ph_idx_to_word_idx

    def set_in_format(self, in_format):
        self.in_format = in_format

    def get_dataset(self, wav_paths):
        # dataset is a pandas dataframe with columns: wav_path, ph_seq, word_seq, ph_idx_to_word_idx
        dataset = []
        for wav_path in wav_paths:
            try:
                if wav_path.with_suffix("." + self.in_format).exists():
                    with open(wav_path.with_suffix("." + self.in_format), "r", encoding="utf-8") as f:
                        lab_text = f.read().strip()
                    ph_seq, word_seq, ph_idx_to_word_idx = self(lab_text)
                    dataset.append((wav_path, ph_seq, word_seq, ph_idx_to_word_idx))
            except Exception as e:
                e.args = (f" Error when processing {wav_path}: {e} ",)
                raise e
        dataset = pd.DataFrame(
            dataset, columns=["wav_path", "ph_seq", "word_seq", "ph_idx_to_word_idx"]
        )
        dataset = DataFrameDataset(dataset)
        return dataset
