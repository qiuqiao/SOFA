import os
import pathlib

import click
import lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import MixedDataset, WeightedBinningAudioBatchSampler, collate_fn
from modules.task.forced_alignment import LitForcedAlignmentTask


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=str,
    default="configs/train_config.yaml",
    show_default=True,
    help="training config path",
)
@click.option(
    "--data_folder",
    "-d",
    type=str,
    default="data",
    show_default=True,
    help="data folder path",
)
@click.option(
    "--pretrained_model_path",
    "-p",
    type=str,
    default=None,
    show_default=True,
    help="pretrained model path. if None, training from scratch",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    default=False,
    show_default=True,
    help="resume training from checkpoint",
)
def main(config_path: str, data_folder: str, pretrained_model_path, resume):
    data_folder = pathlib.Path(data_folder)
    os.environ[
        "TORCH_CUDNN_V8_API_ENABLED"
    ] = "1"  # Prevent unacceptable slowdowns when using 16 precision

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(data_folder / "binary" / "vocab.yaml") as f:
        vocab = yaml.safe_load(f)
    vocab_text = yaml.safe_dump(vocab)

    with open(data_folder / "binary" / "global_config.yaml") as f:
        config_global = yaml.safe_load(f)
    config.update(config_global)

    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    pl.seed_everything(config["random_seed"], workers=True)

    # define dataset
    train_dataset = MixedDataset(
        config["data_augmentation_size"], data_folder / "binary", prefix="train"
    )
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
    lightning_alignment_model = LitForcedAlignmentTask(
        vocab_text,
        config["model"],
        config["melspec_config"],
        config["optimizer_config"],
        config["loss_config"],
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
        max_steps=config["optimizer_config"]["total_steps"],
    )

    ckpt_path = None
    if pretrained_model_path is not None:
        # use pretrained model TODO: load pretrained model
        pretrained = LitForcedAlignmentTask.load_from_checkpoint(pretrained_model_path)
        lightning_alignment_model.load_pretrained(pretrained)
    elif resume:
        # resume training state
        ckpt_path_list = (pathlib.Path("ckpt") / config["model_name"]).rglob("*.ckpt")
        ckpt_path_list = sorted(
            ckpt_path_list, key=lambda x: int(x.stem.split("step=")[-1]), reverse=True
        )
        ckpt_path = str(ckpt_path_list[0]) if len(ckpt_path_list) > 0 else None

    # start training
    trainer.fit(
        model=lightning_alignment_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=ckpt_path,
    )

    # Discard the optimizer and save
    trainer.save_checkpoint(
        str(pathlib.Path("ckpt") / config["model_name"]) + ".ckpt", weights_only=True
    )


if __name__ == "__main__":
    main()
