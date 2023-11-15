import os
import pathlib
from dataset import MixedDataset, collate_fn, WeightedBinningAudioBatchSampler
from torch.utils.data import DataLoader
import lightning as pl
import yaml
import torch
import click
from modules.model.lightning_model import LitForcedAlignmentModel


@click.command()
@click.option("--config_path", "-c", type=str, default="configs/train_config.yaml", show_default=True,
              help="training config path")
@click.option("--data_folder", "-d", type=str, default="data", show_default=True, help="data folder path")
def main(config_path: str, data_folder: str):
    data_folder = pathlib.Path(data_folder)
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(data_folder / "binary" / "vocab.yaml") as f:
        vocab_text = f.read()
    with open(data_folder / "binary" / "global_config.yaml") as f:
        config_global = yaml.safe_load(f)
    config.update(config_global)

    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    pl.seed_everything(config["random_seed"], workers=True)

    # define dataset
    train_dataset = MixedDataset(config["data_augmentation_size"], data_folder / "binary", prefix="train")
    train_sampler = WeightedBinningAudioBatchSampler(
        train_dataset.get_label_types(),
        train_dataset.get_wav_lengths(),
        config["oversampling_weights"],
        config["batch_max_length"] / (2 if config["data_augmentation_size"] > 0 else 1),
        config["binning_length"],
        config["drop_last"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config["dataloader_workers"],
    )

    valid_dataset = MixedDataset(0, data_folder / "binary", prefix="valid")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["dataloader_workers"],
    )

    # model
    lightning_alignment_model = LitForcedAlignmentModel(
        vocab_text,
        config["melspec_config"],
        config["melspec_config"]["n_mels"],
        config["max_frame_num"],
        config["learning_rate"],
        config["weight_decay"],
        config["hidden_dims"],
        config["init_type"],
        config["label_smoothing"],
        config["lr_schedule"],
        config["losses_schedules"],
        config["data_augmentation_size"] > 0,
    )

    # trainer
    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        default_root_dir=str(pathlib.Path("ckpt") / config["model_name"]),
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=None,
        max_epochs=-1,
        max_steps=config["max_steps"],
    )
    # resume training state
    ckpt_path_list = (pathlib.Path("ckpt") / config["model_name"]).rglob("*.ckpt")
    ckpt_path_list = sorted(ckpt_path_list, key=lambda x: int(x.stem.split("step=")[-1]), reverse=True)

    # start training
    trainer.fit(
        model=lightning_alignment_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=str(ckpt_path_list[0]) if len(ckpt_path_list) > 0 else None
    )


if __name__ == "__main__":
    main()
