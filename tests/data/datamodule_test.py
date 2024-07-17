from src.torchsofa.data import MixedDataModule


class TestDatamodule:
    def test_prepare_data(self):
        datamodule = MixedDataModule(preprocess=True, data_dir="data/")
        datamodule.prepare_data()

    def test_setup_fit(self):
        datamodule = MixedDataModule(preprocess=False, data_dir="data/")
        datamodule.prepare_data()
        datamodule.setup("fit")


if __name__ == "__main__":
    TestDatamodule().test_setup_fit()
