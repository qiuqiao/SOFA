import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from modules.model.forced_aligner_model import ForcedAlignmentModel
import pathlib
from dataset import MixedDataset, collate_fn
from torch.utils.data import DataLoader
import lightning as pl
import yaml
from modules.loss.GHMLoss import GHMLoss, CTCGHMLoss
from modules.loss.BinaryEMDLoss import BinaryEMDLoss
from modules.model.forced_aligner_model import ForcedAlignmentModel
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from modules.scheduler.gaussian_ramp_up_scheduler import GaussianRampUpScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import click
from einops import rearrange, repeat
import math


class LitForcedAlignmentModel(pl.LightningModule):
    def __init__(self, config, vocab, model, init_type="kaiming_normal"):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        # read configs
        self.config_train = config["train"]
        self.config = config["global"]
        self.vocab = vocab

        # define model
        self.model = model

        # define loss fn
        self.ph_frame_GHM_loss_fn = GHMLoss(self.vocab["<vocab_size>"], 10, 0.999,
                                            label_smoothing=self.config["label_smoothing"], )
        self.ph_edge_GHM_loss_fn = GHMLoss(2, 10, 0.999999, label_smoothing=self.config["label_smoothing"])
        self.EMD_loss_fn = BinaryEMDLoss()
        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_loss_fn = CTCGHMLoss(alpha=0.999)

        self.init_type = init_type
        self.apply(self.init_weights)

    def init_weights(self, m):
        if self.init_type == "xavier_uniform":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.)
        elif self.init_type == "xavier_normal":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.)
        elif self.init_type == "kaiming_normal":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)
        elif self.init_type == "kaiming_uniform":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.)

    def _get_loss(
            self,
            ph_frame_pred,  # (B, T, vocab_size)
            ph_edge_pred,  # (B, T, 2)
            ctc_pred,  # (B, T, vocab_size)
            ph_frame,  # (B, T)
            ph_edge,  # (B, 2, T)
            ph_seq,  # (sum of ph_seq_lengths)
            ph_seq_lengths,  # (B)
            input_feature_lengths,  # (B)
            label_type,  # (B)
    ):
        # drop according to label_type
        ph_frame_pred = ph_frame_pred[label_type >= 2, :, :]
        ph_frame = ph_frame[label_type >= 2, :]
        ph_edge = ph_edge[label_type >= 2, :, :]
        ph_edge_pred = ph_edge_pred[label_type >= 2, :, :]
        input_lengths_strong = input_feature_lengths[label_type >= 2]
        ctc_pred = ctc_pred[label_type >= 1, :, :]
        input_lengths_weak = input_feature_lengths[label_type >= 1]

        # calculate mask matrix
        mask = torch.arange(ph_frame_pred.size(1)).to(self.device)
        mask = repeat(mask, "T -> B T", B=ph_frame_pred.shape[0])
        mask = mask >= input_lengths_strong.unsqueeze(1)
        mask = mask.to(torch.float32)  # (B, T)

        # ph_frame_loss
        ph_frame_pred = rearrange(ph_frame_pred, "B T C -> B C T")
        ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(ph_frame_pred, ph_frame, mask)

        # ph_edge loss
        ph_edge_pred = rearrange(ph_edge_pred, "B T C -> B C T")
        edge_prob = nn.functional.softmax(ph_edge_pred, dim=1)[:, 0, :]

        ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(ph_edge_pred, ph_edge, mask)

        ph_edge_EMD_loss = self.EMD_loss_fn(edge_prob, ph_edge[:, 0, :])

        edge_prob_diff = torch.diff(edge_prob, 1, dim=-1)
        edge_gt_diff = torch.diff(ph_edge[:, 0, :], 1, dim=-1)
        edge_diff_mask = (edge_gt_diff != 0).to(torch.float32)
        ph_edge_diff_loss = self.MSE_loss_fn(edge_prob_diff * edge_diff_mask, edge_gt_diff * edge_diff_mask)

        ph_edge_loss = ph_edge_GHM_loss + 0.01 * ph_edge_EMD_loss + ph_edge_diff_loss

        # ctc loss
        log_probs_pred = torch.nn.functional.log_softmax(ctc_pred, dim=-1)
        log_probs_pred = rearrange(log_probs_pred, "B T C -> T B C")
        ctc_loss = 0.1 * self.CTC_loss_fn(log_probs_pred, ph_seq, input_lengths_weak, ph_seq_lengths)

        # TODO: self supervised loss

        # total loss
        # TODO: 根据config来设置loss的权重
        loss = ph_frame_GHM_loss + ph_edge_loss + ctc_loss
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (
            input_feature,  # (B, n_mels, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (sum of ph_seq_lengths)
            ph_seq_lengths,  # (B)
            ph_edge,  # (B, 2, T)
            ph_frame,  # (B, T)
            label_type,  # (B)
        ) = batch

        (
            ph_frame_pred,  # (B, T, vocab_size)
            ph_edge_pred,  # (B, T, 2)
            ctc_pred,  # (B, T, vocab_size)
        ) = self.model(input_feature.transpose(1, 2))

        loss = self._get_loss(
            ph_frame_pred,
            ph_edge_pred,
            ctc_pred,
            ph_frame,
            ph_edge,
            ph_seq,
            ph_seq_lengths,
            input_feature_lengths,
            label_type
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config_train["learning_rate"],
            weight_decay=self.config_train["weight_decay"],
        )
        return optimizer


@click.command()
@click.option("--config_path", "-c", type=str, default="configs/config.yaml", show_default=True, help="config path")
def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(pathlib.Path(config["global"]["binary_data_folder"]) / "vocab.yaml") as f:
        vocab = yaml.safe_load(f)

    # define dataset
    train_dataset = MixedDataset(config["global"]["binary_data_folder"], prefix="train")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataset = MixedDataset(config["global"]["binary_data_folder"], prefix="valid")
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # train model
    lightning_alignment_model = LitForcedAlignmentModel(
        config,
        vocab,
        ForcedAlignmentModel(config["global"]["n_mels"],
                             vocab["<vocab_size>"],
                             hidden_dims=64,
                             max_seq_len=config["global"]["max_timestep"] + 32
                             )
    )

    trainer = pl.Trainer(accelerator=config["train"]["accelerator"],
                         gradient_clip_val=config["train"]["gradient_clip_val"],
                         gradient_clip_algorithm=config["train"]["gradient_clip_algorithm"])
    trainer.fit(model=lightning_alignment_model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    main()
