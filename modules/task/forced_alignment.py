from typing import Any

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module
import yaml
from einops import repeat

from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling
from modules.loss.forced_alignment_loss import ForcedAlignmentLoss
from modules.utils.feature_extraction import FeatureExtractor
from modules.utils.load_wav import load_wav
from modules.utils.plot import plot_for_valid


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
        self.get_feature = None
        self.backbone = UNetBackbone(
            melspec_config["n_mels"] + 3,
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
            model_config["hidden_dims"], self.vocab["<vocab_size>"] + 2
        )
        self.melspec_config = melspec_config  # Required for inference
        self.optimizer_config = optimizer_config

        self.pseudo_label_ratio = loss_config["function"]["pseudo_label_ratio"]
        self.pseudo_label_auto_theshold = 0.5

        # loss function
        self.loss = ForcedAlignmentLoss(
            loss_config,
            optimizer_config["total_steps"],
            self.vocab["<vocab_size>"],
            data_augmentation_enabled,
        )

        # validation_step_outputs
        self.validation_step_outputs = {"losses": []}

        self.inference_mode = "force"

    # ==training=============================
    def load_pretrained(self, pretrained_model):
        self.backbone = pretrained_model.backbone
        if self.vocab["<vocab_size>"] == pretrained_model.vocab["<vocab_size>"]:
            self.head = pretrained_model.head
        else:
            self.head = nn.Linear(
                self.backbone.output_dims, self.vocab["<vocab_size>"] + 2
            )

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

    def on_train_start(self):
        self.loss.resume(self.global_step)
        self.loss.to(self.device)

    def training_step(self, batch, batch_idx):
        # try:
        (
            input_feature,  # (B, input_dims, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            ph_frame,  # (B, T)
            ph_mask,  # (B vocab_size)
            label_type,  # (B)
        ) = batch
        (
            ph_frame_logits,  # (B, T, vocab_size)
            ctc_logits,  # (B, T, vocab_size)
        ) = self.forward(input_feature.transpose(1, 2))

        total_loss, losses_dict, schedulers_dict = self.loss._get_loss(
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

        self.loss._losses_schedulers_step()

        log_dict = {"train_" + k: v for k, v in losses_dict.items()}
        log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        log_dict.update(schedulers_dict)
        self.log_dict(log_dict)

        return total_loss
        # except Exception as e:
        #     print(f"Error: {e}. skip this batch.")
        #     return torch.tensor(torch.nan, requires_grad=True).to(self.device)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        h = self.backbone(*args, **kwargs)
        logits = self.head(h)
        ph_frame_logits = logits[:, :, 2:]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)
        return ph_frame_logits, ctc_logits

    # ==validation===========================
    def on_validation_start(self):
        self.on_train_start()

    def validation_step(self, batch, batch_idx):
        (
            input_feature,  # (B, input_dims, T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
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
        _, _, _, _, _, ctc, fig = self._infer_once(
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
            ph_frame_logits,  # (B, T, vocab_size)
            ctc_logits,  # (B, T, vocab_size)
        ) = self.forward(input_feature.transpose(1, 2))

        total_loss, losses_dict, schedulers_dict = self.loss._get_loss(
            ph_frame_logits,
            ctc_logits,
            ph_frame,
            ph_seq,
            ph_seq_lengths,
            ph_mask,
            input_feature_lengths,
            label_type,
            valid=True,
        )

        self.validation_step_outputs["losses"].append(losses_dict)

    def on_validation_epoch_end(self):
        losses_dict_sum = {}
        losses_dict_times = {}
        for losses_dict in self.validation_step_outputs["losses"]:
            for k, v in losses_dict.items():
                if k in losses_dict_sum:
                    losses_dict_sum[k] += v
                    losses_dict_times[k] += 1
                else:
                    losses_dict_sum[k] = v
                    losses_dict_times[k] = 1
        losses_dict_mean = {
            k: v / losses_dict_times[k] for k, v in losses_dict_sum.items()
        }

        self.log_dict({f"valid_{k}": v for k, v in losses_dict_mean.items()})

    # ==predict==============================
    def set_inference_mode(self, mode):
        self.inference_mode = mode

    def on_predict_start(self):
        if self.get_feature is None:
            self.get_feature = FeatureExtractor(**self.melspec_config)

    def predict_step(self, batch, batch_idx):
        try:
            wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
            waveform = load_wav(
                wav_path, self.device, self.melspec_config["sample_rate"]
            )
            wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
            input_feature = self.get_feature(waveform).detach().unsqueeze(0)
            input_feature = (input_feature - input_feature.mean()) / input_feature.std()
            input_feature = repeat(
                input_feature,
                "B C T -> B C (T N)",
                N=self.melspec_config["scale_factor"],
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
                input_feature, ph_seq, word_seq, ph_idx_to_word_idx, False, False
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

    def _infer_once(
        self,
        input_feature,
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
            ) = self.forward(input_feature.transpose(1, 2))

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
            np.concatenate(
                [
                    ph_time_int_pred.astype("float32"),
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
                "input_feature": input_feature.cpu().numpy(),
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

    def _decode(self, ph_seq_id, ph_prob_log):
        # ph_seq_id: (S)
        # ph_prob_log: (T, vocab_size)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)
        # not_SP_num = (ph_seq_id > 0).sum()
        prob_log = ph_prob_log[:, ph_seq_id]

        # init
        curr_ph_max_prob_log = np.zeros(S) - np.inf
        dp = np.zeros([T, S]).astype("float32") - np.inf  # (T, S)
        backtrack_s = np.zeros_like(dp).astype("int32") - 1
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
        for t in range(1, T):
            # [t-1,s] -> [t,s]
            prob1 = dp[t - 1, :] + prob_log[t, :]
            # [t-1,s-1] -> [t,s]
            prob2 = (
                dp[t - 1, :-1] + prob_log[t, :-1] + curr_ph_max_prob_log[:-1] * (T / S)
            )
            prob2 = np.pad(prob2, (1, 0), "constant", constant_values=-np.inf)
            # [t-1,s-2] -> [t,s]
            prob3 = (
                dp[t - 1, :-2] + prob_log[t, :-2] + curr_ph_max_prob_log[:-2] * (T / S)
            )
            prob3[ph_seq_id[1:-1] != 0] = -np.inf  # 不能跳过音素，可以跳过SP
            prob3 = np.pad(
                prob3, (prob3_pad_len, 0), "constant", constant_values=-np.inf
            )

            backtrack_s[t, :] = np.argmax(np.stack([prob1, prob2, prob3]), axis=0)
            curr_ph_max_prob_log[backtrack_s[t, :] == 0] = np.max(
                np.stack(
                    [
                        curr_ph_max_prob_log[backtrack_s[t, :] == 0],
                        prob_log[t, backtrack_s[t, :] == 0],
                    ]
                ),
                axis=0,
            )
            curr_ph_max_prob_log[backtrack_s[t, :] > 0] = prob_log[
                t, backtrack_s[t, :] > 0
            ]
            curr_ph_max_prob_log = curr_ph_max_prob_log * (ph_seq_id > 0)
            dp[t, :] = np.max(np.stack([prob1, prob2, prob3]), axis=0)

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
                # np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
                frame_confidence,
                1,
                prepend=0,
            )
        )

        return (
            np.array(ph_idx_seq),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )
