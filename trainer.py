import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import pathlib
from dataset import BinaryDataset, collate_fn
from torch.utils.data import DataLoader
from modules import loss


class AlignmentModelTrainer:
    def __init__(self, config: dict, vocab: dict):
        with open(
            "ckpt" / pathlib.Path(config["global"]["model_name"]) / "config.yaml", "w"
        ):
            yaml.dump(config)

        with open(
            "ckpt" / pathlib.Path(config["global"]["model_name"]) / "vocab.yaml", "w"
        ):
            yaml.dump(vocab)

        self.config = config["global"]
        self.vocab = vocab

        self.fabric = pl.Fabric(precision=config["train"]["precision"])
        if config["train"]["num_devices"] > 1:
            self.fabric.launch()
        # seed everything
        self.fabric.seed_everything(self.config["random_seed"])

        # dataset
        train_dataset = BinaryDataset(
            binary_data_folder=self.config.binary_data_folder, prefix="train"
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )
        train_dataiter = iter(train_dataloader)

        valid_dataset = BinaryDataset(
            binary_data_folder=self.config.binary_data_folder, prefix="valid"
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
        )

        # model and pretrained model
        # # model.train()

        # loss function
        seg_GHM_loss_fn = loss.GHMLoss(
            config["global"]["device"],
            vocab["<vocab_size>"],
            num_prob_bins=10,
            alpha=0.999,
            label_smoothing=config.label_smoothing,
        )
        edge_GHM_loss_fn = loss.GHMLoss(
            config["global"]["device"],
            2,
            num_prob_bins=5,
            alpha=0.999999,
            label_smoothing=0.0,
            enable_prob_input=True,
        )
        EMD_loss_fn = loss.BinaryEMDLoss()
        MSE_loss_fn = nn.MSELoss()
        CTC_loss_fn = nn.CTCLoss()
        # scheduler
        # optimizer
        # train

        # get data
        # forward
        # calculate loss
        # log
        # sum up losses
        # backward and update
        # log

        # # valid
        # # test

    def train(self):
        pass

    def valid(self):
        pass

    def test(self):
        # model.eval()
        # model.train()
        pass
