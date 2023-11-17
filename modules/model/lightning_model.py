from typing import Any

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module
import yaml
from einops import rearrange, repeat

import modules.scheduler as scheduler_module
from modules.loss.BinaryEMDLoss import BinaryEMDLoss
from modules.loss.GHMLoss import CTCGHMLoss, GHMLoss
from modules.model.forced_aligner_model import ForcedAlignmentModel
from modules.utils.get_melspec import MelSpecExtractor
from modules.utils.load_wav import load_wav
from modules.utils.plot import plot_for_valid


class LitForcedAlignmentModel(pl.LightningModule):
    def __init__(
        self,
        vocab_text,
        melspec_config,
        input_feature_dims,
        max_frame_num,
        learning_rate,
        weight_decay,
        hidden_dims,
        init_type,
        label_smoothing,
        lr_schedule,
        losses_schedules,
        data_augmentation_enabled,
        pseudo_label_ratio,
    ):
        super().__init__()
        # vocab
        self.vocab = yaml.safe_load(vocab_text)

        # hparams
        self.save_hyperparameters()
        self.melspec_config = (
            melspec_config  # Required for inference, but not for training
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dims = hidden_dims
        self.init_type = init_type
        self.label_smoothing = label_smoothing
        self.lr_schedule = lr_schedule
        self.pseudo_label_ratio = pseudo_label_ratio
        self.pseudo_label_auto_theshold = 0.5

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ph_edge_GHM_loss",
            "ph_edge_EMD_loss",
            "ph_edge_diff_loss",
            "ctc_GHM_loss",
            "consistency_loss",
            "pseudo_label_loss",
            "total_loss",
        ]
        self.losses_weights = []
        for k in self.losses_names[:-1]:
            self.losses_weights.append(losses_schedules[k]["weight"])
        self.losses_weights = torch.tensor(self.losses_weights)

        self.losses_schedulers = []
        for k in self.losses_names[:-1]:
            scheduler_type = losses_schedules[k]["scheduler"]["type"]
            scheduler_kwargs = losses_schedules[k]["scheduler"]["kwargs"]
            if scheduler_type is None:
                self.losses_schedulers.append(
                    getattr(scheduler_module, "NoneScheduler")()
                )
            else:
                self.losses_schedulers.append(
                    getattr(scheduler_module, scheduler_type)(**scheduler_kwargs)
                )
        self.data_augmentation_enabled = data_augmentation_enabled

        # model
        self.model = ForcedAlignmentModel(
            input_feature_dims,
            self.vocab["<vocab_size>"],
            hidden_dims=self.hidden_dims,
            max_seq_len=max_frame_num,
        )

        # loss function
        self.ph_frame_GHM_loss_fn = GHMLoss(
            self.vocab["<vocab_size>"],
            10,
            0.999,
            label_smoothing=self.label_smoothing,
        )
        self.pseudo_label_GHM_loss_fn = GHMLoss(
            self.vocab["<vocab_size>"],
            10,
            0.999,
            label_smoothing=self.label_smoothing,
        )
        self.ph_edge_GHM_loss_fn = GHMLoss(
            2, 10, 0.999999, label_smoothing=self.label_smoothing
        )
        self.EMD_loss_fn = BinaryEMDLoss()
        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_GHM_loss_fn = CTCGHMLoss(alpha=0.999)

        # init weights
        self.apply(self.init_weights)

        # get_melspec
        self.get_melspec = None

        # validation_step_outputs
        self.validation_step_outputs = {"losses": []}

        self.inference_mode = "force"

    def on_validation_start(self):
        self.on_train_start()

    def on_train_start(self):
        # resume loss schedulers
        for scheduler in self.losses_schedulers:
            scheduler.resume(self.global_step)
        self.losses_weights = self.losses_weights.to(self.device)

    def _losses_schedulers_step(self):
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(
            self.device
        )

    def init_weights(self, m):
        if self.init_type == "xavier_uniform":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        elif self.init_type == "xavier_normal":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)
        elif self.init_type == "kaiming_normal":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
        elif self.init_type == "kaiming_uniform":
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def _decode(self, ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (T)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)

        edge_prob_log = np.log(edge_prob + 1e-6).astype("float64")
        not_edge_prob_log = np.log(1 - edge_prob + 1e-6).astype("float64")
        # 乘上is_phoneme正确分类的概率 TODO: enable this
        # ph_prob_log[:, 0] += ph_prob_log[:, 0]
        # ph_prob_log[:, 1:] += 1 / ph_prob_log[:, [0]]

        # init
        dp = np.zeros([T, S]).astype("float64") - np.inf  # (T, S)
        backtrack_s = np.zeros_like(dp).astype("int32") - 1
        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        if self.inference_mode == "force":
            dp[0, 0] = ph_prob_log[0, ph_seq_id[0]]
            if ph_seq_id[0] == 0:
                dp[0, 1] = ph_prob_log[0, ph_seq_id[1]]
        # 如果mode==match，可以从任意音素开始
        elif self.inference_mode == "match":
            for i, ph_id in enumerate(ph_seq_id):
                dp[0, i] = ph_prob_log[0, ph_id]

        # forward
        for t in range(1, T):
            # [t-1,s] -> [t,s]
            prob1 = dp[t - 1, :] + ph_prob_log[t, ph_seq_id[:]] + not_edge_prob_log[t]
            # [t-1,s-1] -> [t,s]
            prob2 = dp[t - 1, :-1] + ph_prob_log[t, ph_seq_id[:-1]] + edge_prob_log[t]
            prob2 = np.pad(prob2, (1, 0), "constant", constant_values=-np.inf)
            # [t-1,s-2] -> [t,s]
            prob3 = dp[t - 1, :-2] + ph_prob_log[t, ph_seq_id[:-2]] + edge_prob_log[t]
            prob3[ph_seq_id[1:-1] != 0] = -np.inf  # 不能跳过音素，可以跳过SP
            prob3 = np.pad(prob3, (2, 0), "constant", constant_values=-np.inf)

            backtrack_s[t, :] = np.argmax(np.stack([prob1, prob2, prob3]), axis=0)
            dp[t, :] = np.max(np.stack([prob1, prob2, prob3]), axis=0)

        # backward
        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []
        # 如果mode==forced，只能从最后一个音素或者SP结束
        if self.inference_mode == "force":
            if dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
                s = S - 2
            else:
                s = S - 1
        # 如果mode==match，可以从任意音素结束
        elif self.inference_mode == "match":
            s = np.argmax(dp[-1, :])
        else:
            raise ValueError("inference_mode must be 'force' or 'match'")

        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[t, s] >= 0 or t == 0
            frame_confidence.append(dp[t, s])
            if backtrack_s[t, s] != 0:
                ph_idx_seq.append(s)
                ph_time_int.append(t)
                s -= backtrack_s[t, s]
        ph_idx_seq.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(
            np.diff(
                np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
            )
        )

        return (
            np.array(ph_idx_seq),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )

    def _infer_once(
        self,
        melspec,
        ph_seq,
        word_seq=None,
        ph_idx_to_word_idx=None,
        return_ctc=False,
        return_plot=False,
    ):
        ph_seq_id = np.array([self.vocab[ph] for ph in ph_seq])
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        # forward
        with torch.no_grad():
            (
                ph_frame_pred,  # (B, T, vocab_size)
                ph_edge_pred,  # (B, T, 2)
                ctc_pred,  # (B, T, vocab_size)
            ) = self.forward(melspec.transpose(1, 2))

        ph_frame_pred = ph_frame_pred.squeeze(0)  # (T, vocab_size)
        ph_edge_pred = ph_edge_pred.squeeze(0)  # (T, 2)
        ctc_pred = ctc_pred.squeeze(0)  # (T, vocab_size)

        T, vocab_size = ph_frame_pred.shape

        # decode
        edge = torch.softmax(ph_edge_pred, dim=-1).cpu().numpy().astype("float64")
        edge_diff = np.pad(np.diff(edge[:, 0]), (1, 0), "constant", constant_values=0)
        edge_prob = np.pad(
            edge[1:, 0] + edge[:-1, 0],
            (1, 0),
            "constant",
            constant_values=edge[0, 0] * 2,
        ).clip(0, 1)

        ph_prob_log = (
            torch.log_softmax(ph_frame_pred, dim=-1).cpu().numpy().astype("float64")
        )
        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            ph_prob_log,
            edge_prob,
        )

        # postprocess
        frame_length = (
            self.melspec_config["hop_length"] / self.melspec_config["sample_rate"]
        )
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = frame_length * (
            np.concatenate(
                [
                    ph_time_int_pred.astype("float64") + ph_time_fractional,
                    [T],
                ]
            )
        )
        ph_intervals = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        ph_seq_pred = []
        ph_intervals_pred = []
        word_seq_pred = []
        word_intervals_pred = []

        word_idx_last = -1
        for i, ph_idx in enumerate(ph_idx_seq):
            # ph_idx只能用于两种情况：ph_seq和ph_idx_to_word_idx
            if ph_seq[ph_idx] == "SP":
                continue
            ph_seq_pred.append(ph_seq[ph_idx])
            ph_intervals_pred.append(ph_intervals[i, :])

            word_idx = ph_idx_to_word_idx[ph_idx]
            if word_idx == word_idx_last:
                word_intervals_pred[-1][1] = ph_intervals[i, 1]
            else:
                word_seq_pred.append(word_seq[word_idx])
                word_intervals_pred.append([ph_intervals[i, 0], ph_intervals[i, 1]])
                word_idx_last = word_idx
        ph_seq_pred = np.array(ph_seq_pred)
        ph_intervals_pred = np.array(ph_intervals_pred)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = torch.argmax(ctc_pred, dim=-1).cpu().numpy()
            ctc_index = np.concatenate([[0], ctc])
            ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
            ctc = ctc[ctc_index]
            ctc = np.array([self.vocab[ph] for ph in ctc if ph != 0])

        fig = None
        ph_intervals_pred_int = (
            (ph_intervals_pred / frame_length).round().astype("int32")
        )
        if return_plot:
            ph_frame_id_gt = np.zeros(T)
            for ph, interval in zip(ph_seq_pred, ph_intervals_pred_int):
                ph_frame_id_gt[int(interval[0]) : int(interval[1])] = self.vocab[ph]

            args = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_intervals": ph_intervals_pred_int,
                "frame_confidence": frame_confidence,
                "ph_frame_prob": torch.softmax(ph_frame_pred, dim=-1).cpu().numpy(),
                "ph_frame_id_gt": ph_frame_id_gt,
                "edge_prob": edge_prob,
            }
            fig = plot_for_valid(**args)

        return (
            ph_seq_pred,
            ph_intervals_pred,
            word_seq_pred,
            word_intervals_pred,
            ctc,
            fig,
        )

    def set_inference_mode(self, mode):
        self.inference_mode = mode

    def predict_step(self, batch, batch_idx):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**self.melspec_config)

        waveform = load_wav(wav_path, self.device, self.melspec_config["sample_rate"])
        melspec = self.get_melspec(waveform).unsqueeze(0)
        melspec = (melspec - melspec.mean()) / melspec.std()
        (ph_seq, ph_intervals, word_seq, word_intervals, _, _) = self._infer_once(
            melspec, ph_seq, word_seq, ph_idx_to_word_idx, False, False
        )
        return wav_path, ph_seq, ph_intervals, word_seq, word_intervals

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
        valid=False,
    ):
        full_label_idx = label_type >= 2
        weak_label_idx = label_type >= 1
        T = ph_frame_pred.shape[1]
        ZERO = torch.tensor(0).to(self.device)

        if (full_label_idx).any():
            # drop according to label_type
            ph_frame_full = ph_frame_pred[full_label_idx, :, :]
            ph_frame = ph_frame[full_label_idx, :]
            ph_edge_full = ph_edge_pred[full_label_idx, :, :]
            ph_edge = ph_edge[full_label_idx, :, :]

            # calculate mask matrix
            mask = torch.arange(T).to(self.device)
            mask = repeat(mask, "T -> B T", B=ph_frame_full.shape[0])
            mask = (mask >= input_feature_lengths[full_label_idx].unsqueeze(1)).to(
                ph_frame_full.dtype
            )  # (B, T)

            # ph_frame_loss
            ph_frame_full = rearrange(ph_frame_full, "B T C -> B C T")
            ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(
                ph_frame_full, ph_frame, mask, valid
            )

            # ph_edge loss
            ph_edge_full = rearrange(ph_edge_full, "B T C -> B C T")
            edge_prob = nn.functional.softmax(ph_edge_full, dim=1)[:, 0, :]

            ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(
                ph_edge_full, ph_edge, mask, valid
            )

            ph_edge_EMD_loss = self.EMD_loss_fn(edge_prob, ph_edge[:, 0, :])

            edge_prob_diff = torch.diff(edge_prob, 1, dim=-1)
            edge_gt_diff = torch.diff(ph_edge[:, 0, :], 1, dim=-1)
            edge_diff_mask = (edge_gt_diff != 0).to(ph_frame_full.dtype)
            ph_edge_diff_loss = self.MSE_loss_fn(
                edge_prob_diff * edge_diff_mask, edge_gt_diff * edge_diff_mask
            )

        else:
            ph_frame_GHM_loss = ph_edge_GHM_loss = ZERO
            ph_edge_EMD_loss = ph_edge_diff_loss = ZERO

        if (weak_label_idx).any():
            # drop
            ctc_pred = ctc_pred[weak_label_idx, :, :]
            # ctc loss
            log_probs_pred = torch.nn.functional.log_softmax(ctc_pred, dim=-1)
            log_probs_pred = rearrange(log_probs_pred, "B T C -> T B C")
            ctc_GHM_loss = self.CTC_GHM_loss_fn(
                log_probs_pred,
                ph_seq,
                input_feature_lengths[weak_label_idx],
                ph_seq_lengths,
                valid,
            )
        else:
            ctc_GHM_loss = ZERO

        if not valid and self.data_augmentation_enabled:
            B = ph_frame_pred.shape[0]
            ph_frame_prob_pred = torch.softmax(ph_frame_pred, dim=-1)
            ph_edge_prob_pred = torch.softmax(ph_edge_pred, dim=-1)
            ctc_prob_pred = torch.softmax(ctc_pred, dim=-1)

            # calculate mask matrix
            mask = torch.arange(T).to(self.device)
            mask = repeat(mask, "T -> B T", B=B // 2)
            mask = (mask >= input_feature_lengths[: B // 2].unsqueeze(1)).to(
                torch.bool
            )  # (B//2, T)
            mask_ = mask.unsqueeze(-1).logical_not().float()  # (B//2, T, 1)

            # consistency loss
            consistency_loss = (
                self.MSE_loss_fn(
                    ph_frame_prob_pred[: B // 2, :, :] * mask_,
                    ph_frame_prob_pred[B // 2 :, :, :] * mask_,
                )
                + self.MSE_loss_fn(
                    ph_edge_prob_pred[: B // 2, :, :] * mask_,
                    ph_edge_prob_pred[B // 2 :, :, :] * mask_,
                )
                + self.MSE_loss_fn(
                    ctc_prob_pred[: B // 2, :, :] * mask_,
                    ctc_prob_pred[B // 2 :, :, :] * mask_,
                )
            ) / 3

            # pseudo label loss
            ph_frame_prob_pred = rearrange(ph_frame_prob_pred, "B T C -> B C T")
            pred1_prob, pred1_argmax = torch.max(
                ph_frame_prob_pred[: B // 2, :, :], dim=1
            )
            pred2_prob, pred2_argmax = torch.max(
                ph_frame_prob_pred[B // 2 :, :, :], dim=1
            )
            pseudo_label = pred1_argmax  # (B//2, T)
            pseudo_label_mask = (  # (B//2, T)
                mask
                | (pred1_argmax == pred2_argmax)
                | (((pred1_prob + pred2_prob) / 2) < self.pseudo_label_auto_theshold)
            )
            if (
                pseudo_label_mask.sum() / pseudo_label_mask.numel()
                < self.pseudo_label_ratio
            ):
                self.pseudo_label_auto_theshold += 0.005
            else:
                self.pseudo_label_auto_theshold -= 0.005

            if pseudo_label_mask.logical_not().any():
                pseudo_label_loss = self.pseudo_label_GHM_loss_fn(
                    ph_frame_prob_pred,
                    torch.cat([pseudo_label, pseudo_label], dim=0),
                    torch.cat([pseudo_label_mask, pseudo_label_mask], dim=0),
                    valid,
                )
            else:
                pseudo_label_loss = ZERO
        else:
            consistency_loss = ZERO
            pseudo_label_loss = ZERO

        losses = [
            ph_frame_GHM_loss,
            ph_edge_GHM_loss,
            ph_edge_EMD_loss,
            ph_edge_diff_loss,
            ctc_GHM_loss,
            consistency_loss,
            pseudo_label_loss,
        ]

        return losses

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

        losses = self._get_loss(
            ph_frame_pred,
            ph_edge_pred,
            ctc_pred,
            ph_frame,
            ph_edge,
            ph_seq,
            ph_seq_lengths,
            input_feature_lengths,
            label_type,
            valid=False,
        )

        schedule_weight = self._losses_schedulers_call()
        self._losses_schedulers_step()
        total_loss = (torch.stack(losses) * self.losses_weights * schedule_weight).sum()
        losses.append(total_loss)

        log_dict = {
            f"train_loss/{k}": v for k, v in zip(self.losses_names, losses) if v != 0
        }
        log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        log_dict.update(
            {
                f"scheduler/{k}": v
                for k, v in zip(self.losses_names, schedule_weight)
                if v != 1
            }
        )
        self.log_dict(log_dict)
        return total_loss

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

        ph_seq_g2p = ["SP"]
        for ph in ph_seq.cpu().numpy():
            if ph == 0:
                continue
            ph_seq_g2p.append(self.vocab[ph])
            ph_seq_g2p.append("SP")
        _, _, _, _, ctc, fig = self._infer_once(
            input_feature,
            ph_seq_g2p,
            None,
            None,
            True,
            True,
        )
        self.logger.experiment.add_text(
            f"valid/ctc_predict_{batch_idx}", " ".join(ctc), self.global_step
        )
        self.logger.experiment.add_figure(
            f"valid/plot_{batch_idx}", fig, self.global_step
        )

        (
            ph_frame_pred,  # (B, T, vocab_size)
            ph_edge_pred,  # (B, T, 2)
            ctc_pred,  # (B, T, vocab_size)
        ) = self.forward(input_feature.transpose(1, 2))

        losses = self._get_loss(
            ph_frame_pred,
            ph_edge_pred,
            ctc_pred,
            ph_frame,
            ph_edge,
            ph_seq,
            ph_seq_lengths,
            input_feature_lengths,
            label_type,
            valid=True,
        )

        weights = self._losses_schedulers_call() * self.losses_weights
        total_loss = (torch.stack(losses) * weights).sum()
        losses.append(total_loss)
        losses = torch.stack(losses)

        self.validation_step_outputs["losses"].append(losses)

    def on_validation_epoch_end(self):
        losses = torch.stack(self.validation_step_outputs["losses"], dim=0)
        losses = (losses / ((losses > 0).sum(dim=0, keepdim=True) + 1e-6)).sum(dim=0)
        self.log_dict(
            {f"valid/{k}": v for k, v in zip(self.losses_names, losses) if v != 0}
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": getattr(lr_scheduler_module, self.lr_schedule["type"])(
                optimizer, **self.lr_schedule["kwargs"]
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
