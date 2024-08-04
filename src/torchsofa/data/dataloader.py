import torchaudio
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    def __init__(self, df):
        df.reset_index(drop=True, inplace=True)
        self.df = df

    def get_wav_lengths(self):
        return self.df["wav_length"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # wav, normal_id_seq, normal_inverval_seq, special_id_seq, special_inverval_seq, wav_length, label_type
        row = self.df.iloc[index]

        wav, _ = torchaudio.load(row["wav_path"])
        normal_id_seq = row["normal_id_seq"]
        normal_inverval_seq = row["normal_inverval_seq"]
        special_id_seq = row["special_id_seq"]
        special_inverval_seq = row["special_inverval_seq"]
        wav_length = row["wav_length"]
        label_type = row["label_type"]

        return (
            wav,
            normal_id_seq,
            normal_inverval_seq,
            special_id_seq,
            special_inverval_seq,
            wav_length,
            label_type,
        )
