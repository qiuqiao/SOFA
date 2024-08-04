import warnings
from pathlib import Path

import lightning as L

from .manager import DataManager
from .vocab import Vocab


class MixedDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        random_seed: int = 114514,
        preprocess: bool = True,
        data_dir: str = "data/",
        sample_rate: int = 16000,
        max_length: float = 50.0,
        special_phones: list = ["SP", "AP"],
        ignored_phones: list = ["pau", ""],
        phone_aliases: dict = {"SP": ["<SP>", "<EOS>", "sil", "cl"], "AP": ["<AP>"]},
        valid: dict = {"size": 15, "preferred": ["valid", "test"]},
        loader: dict = {
            "batch_length": 100,
            "randomize_factor": 0.2,
            "num_workers": 16,
        },
    ):
        super().__init__()
        self.random_seed = random_seed
        self.preprocess = preprocess
        self.data_path = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.special_phones = special_phones
        self.ignored_phones = ignored_phones
        self.phone_aliases = phone_aliases
        self.valid = valid
        self.loader = loader

        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} not found")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"{self.data_path} is not a directory")

    def prepare_data(self):
        if not self.preprocess:
            if (self.data_path / "metadata.csv").exists():
                print("Skip preprocessing.")
                return
            else:
                warnings.warn("metadata.csv not found.")

        print("Preprocessing...")

        data = DataManager().prepare(
            self.data_path,
            sample_rate=self.sample_rate,
            max_length=self.max_length,
            labels_from=["transcriptions", "textgrid", "htk"],
        )
        data.save_statistics(self.data_path / "statistics.png")

        data.serialize(self.data_path / "metadata.csv")

    def setup(self, stage: str):
        if stage == "fit":
            # load metadata.csv
            data = DataManager().deserialize(self.data_path / "metadata.csv")
            # generate vocab.csv
            vocab = Vocab(
                all_phones=data.get_phone_set(),
                special_phones=self.special_phones,
                ignored_phones=self.ignored_phones,
                phone_aliases=self.phone_aliases,
            )
            print(vocab.summary())
            vocab.serialize(self.data_path / "vocab.csv")

            data.apply_vocab(vocab)

            data.apply_train_val_split(random_seed=self.random_seed, **self.valid)

            self.train_loader = data.get_train_loader(**self.loader)
            self.valid_loader = data.get_valid_loader(**self.loader)

        # if stage == "test":

        # if stage == "predict":

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    # def test_dataloader(self):
    #     return DataLoader(, batch_size=32)

    # def predict_dataloader(self):
    #     return DataLoader(, batch_size=32)
