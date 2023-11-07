from typing import Any
import lightning as pl
import yaml
from modules.loss.GHMLoss import GHMLoss, CTCGHMLoss
from modules.loss.BinaryEMDLoss import BinaryEMDLoss
from modules.model.forced_aligner_model import ForcedAlignmentModel
import torch.nn as nn
import torch
from einops import rearrange, repeat
from modules.utils.get_melspec import MelSpecExtractor


class LitForcedAlignmentModel(pl.LightningModule):
    def __init__(self,
                 vocab_text,
                 melspec_config,
                 input_feature_dims,
                 max_frame_num,
                 learning_rate,
                 weight_decay,
                 hidden_dims,
                 init_type,
                 label_smoothing,
                 ):
        super().__init__()
        # vocab
        self.vocab = yaml.safe_load(vocab_text)

        # hparams
        self.save_hyperparameters()
        self.melspec_config = melspec_config  # Required for inference, but not for training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dims = hidden_dims
        self.init_type = init_type
        self.label_smoothing = label_smoothing

        self.infer_params = None

        # model
        self.model = ForcedAlignmentModel(input_feature_dims,
                                          self.vocab["<vocab_size>"],
                                          hidden_dims=self.hidden_dims,
                                          max_seq_len=max_frame_num
                                          )

        # loss function
        self.ph_frame_GHM_loss_fn = GHMLoss(self.vocab["<vocab_size>"], 10, 0.999,
                                            label_smoothing=self.label_smoothing, )
        self.ph_edge_GHM_loss_fn = GHMLoss(2, 10, 0.999999, label_smoothing=self.label_smoothing)
        self.EMD_loss_fn = BinaryEMDLoss()
        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_loss_fn = CTCGHMLoss(alpha=0.999)

        # init weights
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

    def set_infer_params(self, kwargs):
        self.infer_params = kwargs

        with open(kwargs["dictionary"], 'r') as f:
            dictionary = f.read().strip().split('\n')
        self.infer_params["dictionary"] = {item.split('\t')[0].strip(): item.split('\t')[1].strip().split(' ')
                                           for item in dictionary}

        self.infer_params["get_melspec"] = MelSpecExtractor(**self.hparams.melspec_config,
                                                            device=self.infer_params["device"])

    def predict_step(self, batch, batch_idx):
        pass

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
        log_values = {}
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
        mask = (mask >= input_lengths_strong.unsqueeze(1)).to(ph_frame_pred.dtype)  # (B, T)

        if ph_frame_pred.shape[0] > 0:
            # ph_frame_loss
            ph_frame_pred = rearrange(ph_frame_pred, "B T C -> B C T")
            ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(ph_frame_pred, ph_frame, mask)
            log_values["ph_frame_GHM_loss"] = ph_frame_GHM_loss.detach()

            # ph_edge loss
            ph_edge_pred = rearrange(ph_edge_pred, "B T C -> B C T")
            edge_prob = nn.functional.softmax(ph_edge_pred, dim=1)[:, 0, :]

            ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(ph_edge_pred, ph_edge, mask)
            log_values["ph_edge_GHM_loss"] = ph_edge_GHM_loss.detach()

            ph_edge_EMD_loss = self.EMD_loss_fn(edge_prob, ph_edge[:, 0, :])
            log_values["ph_edge_EMD_loss"] = ph_edge_EMD_loss.detach()

            edge_prob_diff = torch.diff(edge_prob, 1, dim=-1)
            edge_gt_diff = torch.diff(ph_edge[:, 0, :], 1, dim=-1)
            edge_diff_mask = (edge_gt_diff != 0).to(ph_frame_pred.dtype)
            ph_edge_diff_loss = self.MSE_loss_fn(edge_prob_diff * edge_diff_mask, edge_gt_diff * edge_diff_mask)
            log_values["ph_edge_diff_loss"] = ph_edge_diff_loss.detach()

            ph_edge_loss = ph_edge_GHM_loss + ph_edge_EMD_loss + ph_edge_diff_loss
        else:
            ph_edge_loss = ph_frame_GHM_loss = 0

        if ctc_pred.shape[0] > 0:
            # ctc loss
            log_probs_pred = torch.nn.functional.log_softmax(ctc_pred, dim=-1)
            log_probs_pred = rearrange(log_probs_pred, "B T C -> T B C")
            ctc_loss = self.CTC_loss_fn(log_probs_pred, ph_seq, input_lengths_weak, ph_seq_lengths)
            log_values["ctc_loss"] = ctc_loss.detach()
        else:
            ctc_loss = 0

        # TODO: self supervised loss

        # total loss
        # TODO: 根据config来设置loss的权重
        total_loss = ph_frame_GHM_loss + ph_edge_loss + ctc_loss
        log_values["total_loss"] = total_loss.detach()

        return total_loss, log_values

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

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
        ) = self.forward(input_feature.transpose(1, 2))

        loss, values = self._get_loss(
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
        values = {str("train/" + k): v for k, v in values.items()}
        self.log_dict(values)
        return loss

    def validation_step(self, batch, batch_idx):
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

        val_loss, values = self._get_loss(
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
        values = {str("valid/" + k): v for k, v in values.items()}
        self.log_dict(values)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
