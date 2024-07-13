import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def train(cfg: DictConfig) -> None:
    print("called train")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()
