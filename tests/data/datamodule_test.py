from src.torchsofa.data import MixedDataModule


class TestDatamodule:
    def test_prepare_data(self):
        datamodule = MixedDataModule(preprocess=True, data_dir="data/")
        datamodule.prepare_data()


if __name__ == "__main__":
    TestDatamodule().test_prepare_data()
