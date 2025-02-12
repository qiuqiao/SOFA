from math import sqrt
from typing import Any, Dict

import lightning as pl
import numba
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module
import yaml
from einops import rearrange, repeat

from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling
from modules.loss.GHMLoss import CTCGHMLoss, GHMLoss
from modules.utils.get_melspec import MelSpecExtractor
from modules.utils.load_wav import load_wav
from modules.utils.plot import plot_for_valid

from modules.utils.metrics import (
    Metric,
    VlabelerEditRatio,
    remove_ignored_phonemes,
)
from modules.utils.export_tool import get_textgrid
from modules.utils.label import interval_tier_to_point_tier


@numba.jit
def forward_pass(
    T,
    S,
    prob_log,
    curr_ph_max_prob_log,
    dp,
    backtrack_s,
    ph_seq_id,
    prob3_pad_len,
):
    for t in range(1, T):
        # [t-1,s] -> [t,s]
        prob1 = dp[t - 1, :] + prob_log[t, :]

        prob2 = np.empty(S, dtype=np.float32)
        prob2[0] = -np.inf
        for i in range(1, S):
            prob2[i] = (
                dp[t - 1, i - 1]
                + prob_log[t, i - 1]
                + curr_ph_max_prob_log[i - 1] * (T / S)
            )

        # [t-1,s-2] -> [t,s]
        prob3 = np.empty(S, dtype=np.float32)
        for i in range(prob3_pad_len):
            prob3[i] = -np.inf
        for i in range(prob3_pad_len, S):
            if i - prob3_pad_len + 1 < S - 1 and ph_seq_id[i - prob3_pad_len + 1] != 0:
                prob3[i] = -np.inf
            else:
                prob3[i] = (
                    dp[t - 1, i - prob3_pad_len]
                    + prob_log[t, i - prob3_pad_len]
                    + curr_ph_max_prob_log[i - prob3_pad_len] * (T / S)
                )

        stacked_probs = np.empty((3, S), dtype=np.float32)
        for i in range(S):
            stacked_probs[0, i] = prob1[i]
            stacked_probs[1, i] = prob2[i]
            stacked_probs[2, i] = prob3[i]

        for i in range(S):
            max_idx = 0
            max_val = stacked_probs[0, i]
            for j in range(1, 3):
                if stacked_probs[j, i] > max_val:
                    max_val = stacked_probs[j, i]
                    max_idx = j
            dp[t, i] = max_val
            backtrack_s[t, i] = max_idx

        for i in range(S):
            if backtrack_s[t, i] == 0:
                curr_ph_max_prob_log[i] = max(curr_ph_max_prob_log[i], prob_log[t, i])
            elif backtrack_s[t, i] > 0:
                curr_ph_max_prob_log[i] = prob_log[t, i]

        for i in range(S):
            if ph_seq_id[i] == 0:
                curr_ph_max_prob_log[i] = 0

    return dp, backtrack_s, curr_ph_max_prob_log


class LitForcedAlignmentTask(pl.LightningModule):
    def __init__(
        self,
        vocab_text,
        model_config,
        melspec_config,
        optimizer_config,
        loss_config,
        data_augmentation_enabled,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = yaml.safe_load(vocab_text)

        self.backbone = UNetBackbone(
            melspec_config["n_mels"],
            model_config["hidden_dims"],
            model_config["hidden_dims"],
            ResidualBasicBlock,
            DownSampling,
            UpSampling,
            down_sampling_factor=model_config["down_sampling_factor"],  # 3
            down_sampling_times=model_config["down_sampling_times"],  # 7
            channels_scaleup_factor=model_config["channels_scaleup_factor"],  # 1.5
        )
        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["<vocab_size>"] + 1
        )
        self.melspec_config = melspec_config  # Required for inference
        self.optimizer_config = optimizer_config

        self.losses_names = [
            "ph_frame_GHM_loss",
            "ctc_GHM_loss",
            "consistency_loss",
        ]
        self.losses_weights = torch.tensor(loss_config["losses"]["weights"])

        self.data_augmentation_enabled = data_augmentation_enabled

        # loss function
        self.ph_frame_GHM_loss_fn = GHMLoss(
            self.vocab["<vocab_size>"],
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            loss_config["function"]["label_smoothing"],
        )

        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_GHM_loss_fn = CTCGHMLoss(alpha=1 - 1e-3)

        # get_melspec
        self.get_melspec = None

        # validation_step_outputs
        self.validation_step_outputs = {"losses": []}

        self.inference_mode = "force"

        self.metrics: Dict[str, Metric] = {
            "VlabelerEditRatio10ms": VlabelerEditRatio(move_tolerance=0.01),
            "VlabelerEditRatio20ms": VlabelerEditRatio(move_tolerance=0.02),
            "VlabelerEditRatio50ms": VlabelerEditRatio(move_tolerance=0.05),
        }

    def load_pretrained(self, pretrained_model):
        self.backbone = pretrained_model.backbone
        if self.vocab["<vocab_size>"] == pretrained_model.vocab["<vocab_size>"]:
            self.head = pretrained_model.head
        else:
            self.head = nn.Linear(
                self.backbone.output_dims, self.vocab["<vocab_size>"] + 1
            )

    def on_validation_start(self):
        self.on_train_start()

    def on_train_start(self):
        self.losses_weights = self.losses_weights.to(self.device)

    def _decode(self, ph_seq_id, ph_prob_log):
        # ph_seq_id: (S)
        # ph_prob_log: (T, vocab_size)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)
        # not_SP_num = (ph_seq_id > 0).sum()
        prob_log = ph_prob_log[:, ph_seq_id]

        # init
        curr_ph_max_prob_log = np.full(S, -np.inf)
        dp = np.full((T, S), -np.inf, dtype="float32")  # (T, S)
        backtrack_s = np.full_like(dp, -1, dtype="int32")
        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        if self.inference_mode == "force":
            dp[0, 0] = prob_log[0, 0]
            curr_ph_max_prob_log[0] = prob_log[0, 0]
            if ph_seq_id[0] == 0 and prob_log.shape[-1] > 1:
                dp[0, 1] = prob_log[0, 1]
                curr_ph_max_prob_log[1] = prob_log[0, 1]
        # 如果mode==match，可以从任意音素开始
        elif self.inference_mode == "match":
            for i, ph_id in enumerate(ph_seq_id):
                dp[0, i] = prob_log[0, i]
                curr_ph_max_prob_log[i] = prob_log[0, i]

        # forward
        prob3_pad_len = 2 if S >= 2 else 1
        dp, backtrack_s, curr_ph_max_prob_log = forward_pass(
            T,
            S,
            prob_log,
            curr_ph_max_prob_log,
            dp,
            backtrack_s,
            ph_seq_id,
            prob3_pad_len,
        )

        # backward
        ph_idx_seq = []
        ph_time_int = []
        frame_confidence = []
        # 如果mode==forced，只能从最后一个音素或者SP结束
        if self.inference_mode == "force":
            if S >= 2 and dp[-1, -2] > dp[-1, -1] and ph_seq_id[-1] == 0:
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
        wav_length,
        ph_seq,
        word_seq=None,
        ph_idx_to_word_idx=None,
        return_ctc=False,
        return_plot=False,
    ):
        ph_seq_id = np.array([self.vocab[ph] for ph in ph_seq])
        ph_mask = np.zeros(self.vocab["<vocab_size>"])
        ph_mask[ph_seq_id] = 1
        ph_mask[0] = 1
        ph_mask = torch.from_numpy(ph_mask)
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))

        # forward
        with torch.no_grad():
            (
                ph_frame_logits,  # (B, T, vocab_size)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(melspec.transpose(1, 2))
        if wav_length is not None:
            num_frames = int(
                (
                    (
                        wav_length
                        * self.melspec_config["scale_factor"]
                        * self.melspec_config["sample_rate"]
                        + 0.5
                    )
                )
                / self.melspec_config["hop_length"]
            )
            ph_frame_logits = ph_frame_logits[:, :num_frames, :]
            ctc_logits = ctc_logits[:, :num_frames, :]

        ph_mask = (
            ph_mask.to(ph_frame_logits.device).unsqueeze(0).unsqueeze(0).logical_not()
            * 1e9
        )
        ph_frame_pred = (
            torch.nn.functional.softmax(
                ph_frame_logits.float() - ph_mask.float(), dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )
        ph_prob_log = (
            torch.log_softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype("float32")
        )

        ctc_logits = (
            ctc_logits.float().squeeze(0).cpu().numpy().astype("float32")
        )  # (ctc_logits.squeeze(0) - ph_mask)

        T, vocab_size = ph_frame_pred.shape

        # decode
        (
            ph_idx_seq,
            ph_time_int_pred,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            ph_prob_log,
        )
        total_confidence = np.exp(np.mean(np.log(frame_confidence + 1e-6)) / 3)

        # postprocess
        frame_length = self.melspec_config["hop_length"] / (
            self.melspec_config["sample_rate"] * self.melspec_config["scale_factor"]
        )
        ph_time_pred = frame_length * (
            np.concatenate([ph_time_int_pred.astype("float32"), [T]])
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
        ph_intervals_pred = np.array(ph_intervals_pred).clip(min=0, max=None)
        word_seq_pred = np.array(word_seq_pred)
        word_intervals_pred = np.array(word_intervals_pred).clip(min=0, max=None)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = np.argmax(ctc_logits, axis=-1)
            ctc_index = np.concatenate([[0], ctc])
            ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
            ctc = ctc[ctc_index]
            ctc = np.array([self.vocab[ph] for ph in ctc if ph != 0])

        fig = None
        ph_intervals_pred_int = (
            (ph_intervals_pred / frame_length).round().astype("int32")
        )
        if return_plot:
            ph_idx_frame = np.zeros(T).astype("int32")
            last_ph_idx = 0
            for ph_idx, ph_time in zip(ph_idx_seq, ph_time_int_pred):
                ph_idx_frame[ph_time] += ph_idx - last_ph_idx
                last_ph_idx = ph_idx
            ph_idx_frame = np.cumsum(ph_idx_frame)
            args = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_intervals": ph_intervals_pred_int,
                "frame_confidence": frame_confidence,
                "ph_frame_prob": ph_frame_pred[:, ph_seq_id],
                "ph_frame_id_gt": ph_idx_frame,
            }
            fig = plot_for_valid(**args)

        return (
            ph_seq_pred,
            ph_intervals_pred,
            word_seq_pred,
            word_intervals_pred,
            total_confidence,
            ctc,
            fig,
        )

    def set_inference_mode(self, mode):
        self.inference_mode = mode

    def on_predict_start(self):
        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**self.melspec_config)

    def predict_step(self, batch, batch_idx):
        try:
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
            waveform = load_wav(
                wav_path, self.device, self.melspec_config["sample_rate"]
            )
            wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
            melspec = self.get_melspec(waveform).detach().unsqueeze(0)
            melspec = (melspec - melspec.mean()) / melspec.std()
            melspec = repeat(
                melspec, "B C T -> B C (T N)", N=self.melspec_config["scale_factor"]
            )

            (
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
                confidence,
                _,
                _,
            ) = self._infer_once(
                melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx, False, False
            )

            return (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
        except Exception as e:
            e.args += (f"{str(wav_path)}",)
            raise e

    def _get_full_label_loss(
        self,
        ph_frame_logits,
        ph_frame_gt,
        input_feature_lengths,
        ph_mask,
        valid,
    ):
        T = ph_frame_logits.shape[1]

        # calculate mask matrix
        # (B, T)
        mask = torch.arange(T).to(self.device)
        mask = repeat(mask, "T -> B T", B=ph_frame_logits.shape[0])
        mask = (mask < input_feature_lengths.unsqueeze(1)).to(ph_frame_logits.dtype)

        # ph_frame_loss
        # print(mask.unsqueeze(-1).shape, ph_mask.unsqueeze(1).shape)

        ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(
            ph_frame_logits,
            ph_frame_gt,
            (mask.unsqueeze(-1) * ph_mask.unsqueeze(1)),
            valid,
        )

        return ph_frame_GHM_loss

    def _get_weak_label_loss(
        self,
        ctc_logits,
        ph_mask,
        ph_seq_gt,
        ph_seq_lengths_gt,
        input_feature_lengths,
        valid,
    ):
        ctc_logits = ctc_logits - ph_mask.unsqueeze(1).logical_not().float() * 1e9
        log_probs_pred = nn.functional.log_softmax(ctc_logits, dim=-1)
        # ctc loss
        log_probs_pred = rearrange(log_probs_pred, "B T C -> T B C")
        ctc_GHM_loss = self.CTC_GHM_loss_fn(
            log_probs_pred,
            ph_seq_gt,
            input_feature_lengths,
            ph_seq_lengths_gt,
            valid,
        )

        return ctc_GHM_loss

    def _get_consistency_loss(self, ph_frame_logits, input_feature_lengths):

        output_tensors = torch.nn.functional.sigmoid(ph_frame_logits.float())
        B = output_tensors.shape[0]
        T = output_tensors.shape[1]

        # calculate mask matrix
        # (B//2, T, 1)
        mask = torch.arange(T).to(self.device)
        mask = repeat(mask, "T -> B T", B=B // 2)
        mask = (
            (mask < input_feature_lengths[: B // 2].unsqueeze(1))
            .to(torch.bool)
            .unsqueeze(-1)
        )

        # consistency loss
        consistency_loss = self.MSE_loss_fn(
            output_tensors[: B // 2, :, :] * mask,
            output_tensors[B // 2 :, :, :] * mask,
        )

        return consistency_loss

    def _get_loss(
        self,
        ph_frame_logits,  # (B, T, vocab_size)
        ctc_logits,  # (B, T, vocab_size)
        ph_frame_gt,  # (B, T)
        ph_seq_gt,  # (B S)
        ph_seq_lengths_gt,  # (B)
        ph_mask,  # (B vocab_size)
        input_feature_lengths,  # (B)
        label_type,  # (B)
        valid=False,
    ):
        full_label_idx = label_type >= 2
        weak_label_idx = label_type >= 1
        not_full_label_idx = label_type < 2
        ZERO = torch.tensor(0.0).to(self.device)

        if (full_label_idx).any():
            ph_frame_GHM_loss = self._get_full_label_loss(
                ph_frame_logits[full_label_idx, :, :],
                ph_frame_gt[full_label_idx, :],
                input_feature_lengths[full_label_idx],
                ph_mask[full_label_idx, :],
                valid,
            )
        else:
            ph_frame_GHM_loss = ZERO

        # TODO:这种pack方式无法处理只有batch中的一部分需要计算Loss的情况，改掉
        if (weak_label_idx).any():
            ctc_GHM_loss = ZERO
            ctc_GHM_loss = self._get_weak_label_loss(
                ctc_logits[weak_label_idx, :, :],
                ph_mask[weak_label_idx, :],
                ph_seq_gt[weak_label_idx, :],
                ph_seq_lengths_gt[weak_label_idx],
                input_feature_lengths[weak_label_idx],
                valid,
            )
        else:
            ctc_GHM_loss = ZERO

        if not valid and self.data_augmentation_enabled:
            consistency_loss = self._get_consistency_loss(
                ph_frame_logits, input_feature_lengths
            )
        else:
            consistency_loss = ZERO

        losses = [
            ph_frame_GHM_loss,
            ctc_GHM_loss,
            consistency_loss,
        ]

        return losses

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        h = self.backbone(*args, **kwargs)
        logits = self.head(h)
        ph_frame_logits = logits[:, :, 1:]
        ph_frame_logits = 6 * ph_frame_logits / sqrt(ph_frame_logits.shape[-1])
        ctc_logits = torch.cat([logits[:, :, [0]], logits[:, :, 2:]], dim=-1)
        ctc_logits = 6 * ctc_logits / sqrt(ctc_logits.shape[-1])
        return ph_frame_logits, ctc_logits

    def training_step(self, batch, batch_idx):
        try:
            (
                input_feature,  # (B, n_mels, T)
                input_feature_lengths,  # (B)
                ph_seq,  # (B S)
                ph_seq_lengths,  # (B)
                ph_intervals,  # (B, L, 2)
                ph_frame,  # (B, T)
                ph_mask,  # (B vocab_size)
                label_type,  # (B)
            ) = batch

            (
                ph_frame_logits,  # (B, T, vocab_size)
                ctc_logits,  # (B, T, vocab_size)
            ) = self.forward(input_feature.transpose(1, 2))

            losses = self._get_loss(
                ph_frame_logits,
                ctc_logits,
                ph_frame,
                ph_seq,
                ph_seq_lengths,
                ph_mask,
                input_feature_lengths,
                label_type,
                valid=False,
            )

            log_dict = {
                f"train_loss/{k}": v
                for k, v in zip(self.losses_names, losses)
                if v != 0
            }
            log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log_dict(log_dict)

            losses = (
                torch.stack(losses)
                / (1e-3 + torch.stack(losses).detach())
                * self.losses_weights
            )
            total_loss = (losses).sum()

            return total_loss
        except Exception as e:
            print(f'Error: "{e}." skip this batch.')
            raise e
            return torch.tensor(torch.nan, requires_grad=True).to(self.device)

    def validation_step(self, batch, batch_idx):
        (
            input_feature,  # (B, n_mels, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            ph_intervals,  # (B, L, 2)
            ph_frame,  # (B, T)
            ph_mask,  # (B vocab_size)
            label_type,  # (B)
        ) = batch

        ph_seq_g2p = ["SP"]
        for ph in ph_seq.squeeze(0).cpu().numpy():
            if ph == 0:
                continue
            ph_seq_g2p.append(self.vocab[ph])
            ph_seq_g2p.append("SP")
        (
            ph_seq_pred,
            ph_intervals_pred,
            word_seq_pred,
            word_intervals_pred,
            total_confidence,
            ctc,
            fig,
        ) = self._infer_once(
            input_feature,
            None,
            ph_seq_g2p,
            None,
            None,
            True,
            True,
        )

        # full label metrics
        if label_type.item() == 2:
            # conver to textgird for evaluation

            tg_pred = get_textgrid(
                ph_seq_pred,
                ph_intervals_pred,
                word_seq_pred,
                word_intervals_pred,
            )
            tg_gt = get_textgrid(
                [self.vocab[id.item()] for id in ph_seq.squeeze()],
                ph_intervals.squeeze().T.cpu().numpy(),
                [],
                [],
            )

            tg_pred = interval_tier_to_point_tier(tg_pred[1])
            tg_gt = interval_tier_to_point_tier(tg_gt[1])
            # print("-----")
            # print("pred")
            # for i in tg_pred:
            #     print(i.time, i.mark)
            tg_pred = remove_ignored_phonemes([None, " ", "", "SP", "AP"], tg_pred)
            tg_gt = remove_ignored_phonemes([None, " ", "", "SP", "AP"], tg_gt)

            for key in self.metrics:
                self.metrics[key].update(tg_pred, tg_gt)

        # TODO: weak label metrics
        ## total_confidence
        ## ctc loss

        self.logger.experiment.add_text(
            f"valid/ctc_predict_{batch_idx}", " ".join(ctc), self.global_step
        )
        self.logger.experiment.add_figure(
            f"valid/plot_{batch_idx}", fig, self.global_step
        )

    def on_validation_epoch_end(self):
        d = {f"valid/{k}": v.compute() for k, v in self.metrics.items()}
        self.log_dict(d)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.backbone.parameters(),
                    "lr": self.optimizer_config["lr"]["backbone"],
                },
                {
                    "params": self.head.parameters(),
                    "lr": self.optimizer_config["lr"]["head"],
                },
            ],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.OneCycleLR(
                optimizer,
                max_lr=[
                    self.optimizer_config["lr"]["backbone"],
                    self.optimizer_config["lr"]["head"],
                ],
                total_steps=self.optimizer_config["total_steps"],
            ),
            "interval": "step",
        }

        for k, v in self.optimizer_config["freeze"].items():
            if v:
                getattr(self, k).requires_grad_(False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
