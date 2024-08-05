import hydra
import lightning as pl
from omegaconf import DictConfig, OmegaConf

from torchsofa.data import MixedDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def train_cli(cfg: DictConfig) -> None:
    print("Training...")
    print("config:")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.random_seed)

    datamodule = MixedDataModule(cfg.data)


if __name__ == "__main__":
    train_cli()
