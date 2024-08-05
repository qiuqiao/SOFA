from src.torchsofa.data import MixedDataModule


class TestDatamodule:
    def test_prepare_data(self):
        datamodule = MixedDataModule(preprocess=True, data_dir="data/")
        datamodule.prepare_data()

    def test_setup_fit(self):
        datamodule = MixedDataModule(preprocess=False, data_dir="data/")
        datamodule.prepare_data()
        datamodule.setup("fit")

    def test_dataloader(self):
        datamodule = MixedDataModule(preprocess=False, data_dir="data/")
        datamodule.prepare_data()
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        valid_loader = datamodule.val_dataloader()

        cnt = 0
        for batch in train_loader:
            print(cnt)
            for item in batch:
                print(item.shape)
            cnt += 1
            if cnt == 10:
                break

        cnt = 0
        for batch in valid_loader:
            print(cnt)
            for item in batch:
                print(item.shape)
            cnt += 1
            if cnt == 10:
                break


if __name__ == "__main__":
    TestDatamodule().test_dataloader()
