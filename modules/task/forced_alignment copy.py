import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module
import yaml
from einops import rearrange, repeat

import modules.scheduler as scheduler_module
from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling
from modules.utils.get_melspec import MelSpecExtractor
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

        # model
        self.audio_encoder = UNetBackbone(
            melspec_config["n_mels"],
            model_config["audio_encoder"]["hidden_dims"] + 2,
            model_config["num_embeddings"],
            ResidualBasicBlock,
            DownSampling,
            UpSampling,
            down_sampling_factor=model_config["audio_encoder"]["down_sampling_factor"],
            down_sampling_times=model_config["audio_encoder"]["down_sampling_times"],
            channels_scaleup_factor=model_config["audio_encoder"][
                "channels_scaleup_factor"
            ],
        )
        self.phoneme_encoder = nn.Embedding(
            self.vocab["<vocab_size>"], model_config["num_embeddings"]
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(2, model_config["num_embeddings"], bias=False),
            nn.ReLU(),
        )
        self.parameter = nn.Parameter(torch.ones(2) * -1)

        self.melspec_config = melspec_config  # Required for inference
        self.optimizer_config = optimizer_config

        self.pseudo_label_ratio = loss_config["function"]["pseudo_label_ratio"]
        self.pseudo_label_auto_theshold = 0.5

        self.losses_names = [
            "full_label_loss",
            "ctc_loss",
            "consistency_loss",
        ]
        self.losses_weights = torch.tensor(loss_config["losses"]["weights"])

        self.losses_schedulers = []
        for enabled in loss_config["losses"]["enable_RampUpScheduler"]:
            if enabled:
                self.losses_schedulers.append(
                    scheduler_module.GaussianRampUpScheduler(
                        max_steps=optimizer_config["total_steps"]
                    )
                )
            else:
                self.losses_schedulers.append(scheduler_module.NoneScheduler())
        self.data_augmentation_enabled = data_augmentation_enabled

        # loss functions

        # get_melspec
        self.get_melspec = None

        # validation_step_outputs
        self.validation_step_outputs = {"losses": []}

        self.inference_mode = "force"

    def load_pretrained(self, pretrained_model):
        self.audio_encoder = pretrained_model.audio_encoder
        self.parameter = pretrained_model.parameter
        if pretrained_model.vocab["<vocab_size>"] == self.vocab["<vocab_size>"]:
            self.phoneme_encoder = pretrained_model.phoneme_encoder
        else:
            print("phoneme_encoder not loaded. vocab size mismatch.")

    def set_inference_mode(self, mode):
        self.inference_mode = mode

    def _losses_schedulers_step(self):
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(
            self.device
        )

    def _decode(self, ph_seq_id, alignment_matrix):
        # ph_seq_id: (S)
        # alignment_matrix: (T, S)
        T, S = alignment_matrix.shape
        assert len(ph_seq_id) == S

        # init
        curr_ph_max_logprob = np.zeros(S) - np.inf
        dp = np.zeros([T, S]).astype("float32") - np.inf  # (T, S)
        backtrack_s = np.zeros_like(dp).astype("int32") - 1
        # 如果mode==forced，只能从SP开始或者从第一个音素开始
        if self.inference_mode == "force":
            dp[0, 0] = alignment_matrix[0, 0]
            curr_ph_max_logprob[0] = alignment_matrix[0, 0]
            if ph_seq_id[0] == 0:
                dp[0, 1] = alignment_matrix[0, 1]
                curr_ph_max_logprob[1] = alignment_matrix[0, 1]
        # 如果mode==match，可以从任意音素开始
        elif self.inference_mode == "match":
            dp[0, :] = alignment_matrix[0, :]
            curr_ph_max_logprob[:] = alignment_matrix[0, :]
        else:
            raise ValueError("inference_mode must be 'force' or 'match'")

        # forward
        for t in range(1, T):
            # [t-1,s] -> [t,s]
            prob1 = dp[t - 1, :] + alignment_matrix[t, :]
            # [t-1,s-1] -> [t,s]
            prob2 = (
                dp[t - 1, :-1]
                + alignment_matrix[t, :-1]
                + curr_ph_max_logprob[:-1] * (T / S)
            )
            prob2 = np.pad(prob2, (1, 0), "constant", constant_values=-np.inf)
            # [t-1,s-2] -> [t,s]
            prob3 = (
                dp[t - 1, :-2]
                + alignment_matrix[t, :-2]
                + curr_ph_max_logprob[:-2] * (T / S)
            )
            prob3[ph_seq_id[1:-1] != 0] = -np.inf  # 不能跳过音素，可以跳过SP
            prob3 = np.pad(prob3, (2, 0), "constant", constant_values=-np.inf)

            backtrack_s[t, :] = np.argmax(np.stack([prob1, prob2, prob3]), axis=0)
            curr_ph_max_logprob[backtrack_s[t, :] == 0] = np.max(
                np.stack(
                    [
                        curr_ph_max_logprob[backtrack_s[t, :] == 0],
                        alignment_matrix[t, backtrack_s[t, :] == 0],
                    ]
                ),
                axis=0,
            )
            curr_ph_max_logprob[backtrack_s[t, :] > 0] = alignment_matrix[
                t, backtrack_s[t, :] > 0
            ]
            curr_ph_max_logprob = curr_ph_max_logprob * (ph_seq_id > 0)
            dp[t, :] = np.max(np.stack([prob1, prob2, prob3]), axis=0)

        # backward
        ph_idx_seq = []
        ph_position_idx = []
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
                ph_position_idx.append(t)
                s -= backtrack_s[t, s]
        ph_idx_seq.reverse()
        ph_position_idx.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(
            np.diff(
                np.pad(frame_confidence, (1, 0), "constant", constant_values=0.0), 1
            )
        )

        return (
            np.array(ph_idx_seq),
            np.array(ph_position_idx),
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
        ph_seq_id_without_SP = ph_seq_id[ph_seq_id != 0]
        if word_seq is None:
            word_seq = ph_seq
            ph_idx_to_word_idx = np.arange(len(ph_seq))
        alignment_matrix_index = np.cumsum((ph_seq_id != 0).astype("int32")) * (
            ph_seq_id != 0
        ).astype("int32")
        T = melspec.shape[-1]
        S = len(ph_seq_id_without_SP)

        # forward
        with torch.no_grad():
            audio_embed, phoneme_embed, attn_logits, ctc_logits = self.forward(
                melspec.transpose(1, 2),
                torch.tensor([T]).to(self.device),
                torch.from_numpy(ph_seq_id_without_SP).to(self.device).unsqueeze(0),
                torch.tensor([S]).to(self.device),
            )
        attn_probs = (
            nn.functional.softmax(attn_logits, dim=-1).cpu().numpy()
        )  # (B T S+1)
        attn_logprobs = (
            nn.functional.log_softmax(attn_logits, dim=-1).cpu().numpy()
        )  # (B T S+1)

        alignment_matrix_probs = attn_probs[:, :, alignment_matrix_index]
        alignment_matrix_logprobs = attn_logprobs[:, :, alignment_matrix_index]
        # decode
        (
            ph_idx_seq,
            ph_position_idx,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            alignment_matrix_logprobs.squeeze(0),
        )

        # postprocess
        ph_intervals_idx = np.stack(
            [ph_position_idx[:], np.concatenate([ph_position_idx[1:], [T]])], axis=-1
        )
        frame_length = self.melspec_config["hop_length"] / (
            self.melspec_config["sample_rate"] * self.melspec_config["scale_factor"]
        )
        ph_intervals = ph_intervals_idx * frame_length

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

        ctc = None
        fig = None
        if return_plot:
            ph_idx_frame = np.zeros(T).astype("int32")
            for ph_idx, (start, end) in zip(ph_idx_seq, ph_intervals_idx):
                ph_idx_frame[start:end] = ph_idx
            args = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_intervals": (ph_intervals_pred / frame_length),
                "frame_confidence": frame_confidence,
                "ph_frame_prob": alignment_matrix_probs,
                "ph_frame_id_gt": ph_idx_frame,
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

    def _get_full_label_loss(
        self,
        attn_logits,  # (B T S+1)
        attn_target,  # (B T), item in [0,S]
        input_feature_lengths,  # (B)
        ph_seq_lengths,  # (B)
        valid,
    ):
        B, T, S_ = attn_logits.shape
        # mask
        # (B T)
        feature_len_mask = (
            (
                torch.arange(T).to(self.device).unsqueeze(0)
                < (input_feature_lengths.unsqueeze(1))
            )
            .to(self.device)
            .detach()
        )
        # (B S+1)
        sequence_len_mask = (
            (
                torch.arange(S_).to(self.device).unsqueeze(0)
                < (ph_seq_lengths.unsqueeze(1) + 1)
            )
            .to(self.device)
            .detach()
        )
        # (B T S+1)
        mask_matrix = feature_len_mask.unsqueeze(-1) * sequence_len_mask.unsqueeze(1)
        attn_logits = attn_logits - mask_matrix.logical_not().float() * 1e9

        # calculate loss
        # (B T)
        # TODO: add weight、GHM
        loss = nn.functional.cross_entropy(
            attn_logits.transpose(1, 2),
            attn_target.long(),
            reduction="none",
        )
        # apply mask
        loss = loss * feature_len_mask.float()
        loss = loss.sum() / feature_len_mask.sum()

        return loss

    def _get_weak_label_loss(
        self, attn_logits, input_feature_lengths, ph_seq_lengths, valid
    ):
        log_probs = nn.functional.log_softmax(attn_logits, dim=-1)
        log_probs = rearrange(log_probs, "B T S -> T B S")
        targets = torch.arange(max(ph_seq_lengths)).to(self.device) + 1
        targets = repeat(targets, "S -> B S", B=attn_logits.shape[0])
        targets = targets * (targets < (ph_seq_lengths.unsqueeze(1) + 1)).float()
        targets = targets.long()
        # ctc loss
        # TODO: use ghm loss
        loss = nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths=input_feature_lengths,
            target_lengths=ph_seq_lengths,
            reduction="sum",
        )

        return loss

    def _get_consistency_loss(
        self, audio_embed, input_feature_lengths, ph_seq_lengths, valid
    ):
        B, T, E = audio_embed.shape  # (B T E)
        S = torch.max(ph_seq_lengths)  # ()
        # mask
        # (B T)
        feature_len_mask = torch.arange(T).to(self.device).unsqueeze(0) < (
            input_feature_lengths.unsqueeze(1)
        )
        # (B S)
        sequence_len_mask = torch.arange(S).to(self.device).unsqueeze(0) < (
            ph_seq_lengths.unsqueeze(1)
        )
        # (B T S)
        mask_matrix = feature_len_mask.unsqueeze(-1) * sequence_len_mask.unsqueeze(1)

        # consistency loss
        consistency_loss = (
            audio_embed[: B // 2]
            * audio_embed[B // 2 :]
            * feature_len_mask[B // 2 :].unsqueeze(-1).float()
        ).abs().sum() / (mask_matrix.sum() + 1)
        return consistency_loss

    def _get_loss(
        self,
        audio_embed,  # (B T E)
        phoneme_embed,  # (B S E)
        attn_logits,  # (B T S+1)
        ctc_logits,  # (B T S+1)
        input_feature_lengths,  # (B)
        ph_seq,  # (B S)
        ph_seq_lengths,  # (B)
        attn_target,  # (B T), item in [0,S]
        label_type,  # (B)
        valid=False,
    ):
        full_label_idx = label_type >= 2
        weak_label_idx = label_type >= 1
        # not_full_label_idx = label_type < 2
        ZERO = torch.tensor(0).to(self.device)

        if (full_label_idx).any():
            full_label_loss = self._get_full_label_loss(
                attn_logits[full_label_idx],
                attn_target[full_label_idx],
                input_feature_lengths[full_label_idx],
                ph_seq_lengths[full_label_idx],
                valid,
            )
        else:
            full_label_loss = ZERO

        if (weak_label_idx).any():
            ctc_loss = self._get_weak_label_loss(
                ctc_logits[weak_label_idx],
                input_feature_lengths[weak_label_idx],
                ph_seq_lengths[weak_label_idx],
                valid,
            )
        else:
            ctc_loss = ZERO

        if not valid and self.data_augmentation_enabled:
            consistency_loss = self._get_consistency_loss(
                audio_embed, input_feature_lengths, ph_seq_lengths, valid
            )
            # TODO: add reconstruction loss
        else:
            consistency_loss = ZERO

        if torch.isinf(full_label_loss) or torch.isnan(full_label_loss):
            raise ValueError("full_label_loss is inf or nan")
        if torch.isinf(ctc_loss) or torch.isnan(ctc_loss):
            raise ValueError("ctc_loss is inf or nan")
        if torch.isinf(consistency_loss) or torch.isnan(consistency_loss):
            raise ValueError("consistency_loss is inf or nan")
        losses = [
            full_label_loss,
            ctc_loss,
            consistency_loss,
        ]

        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.audio_encoder.parameters(),
                    "lr": self.optimizer_config["lr"]["audio_encoder"],
                },
                {
                    "params": self.phoneme_encoder.parameters(),
                    "lr": self.optimizer_config["lr"]["phoneme_encoder"],
                },
                {
                    "params": self.parameter,
                    "lr": self.optimizer_config["lr"]["phoneme_encoder"],
                },
                {
                    "params": self.position_encoder.parameters(),
                    "lr": self.optimizer_config["lr"]["phoneme_encoder"],
                },
            ],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.OneCycleLR(
                optimizer,
                max_lr=[
                    self.optimizer_config["lr"]["audio_encoder"],
                    self.optimizer_config["lr"]["phoneme_encoder"],
                    self.optimizer_config["lr"]["phoneme_encoder"],  # TODO: remove this
                    self.optimizer_config["lr"]["phoneme_encoder"],  # TODO: remove this
                ],
                total_steps=self.optimizer_config["total_steps"],
            ),
            "interval": "step",
        }

        for k, v in self.optimizer_config["freeze"].items():
            if v:
                getattr(self, k).requires_grad_(False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, input_feature, input_feature_lengths, ph_seq, ph_seq_lengths):
        audio_embed = self.audio_encoder(input_feature)  # (B T E+2)
        sp_logits = audio_embed[:, :, [0]]  # (B T 1)
        ctc_blank_logits = audio_embed[:, :, [1]]  # (B T 1)
        audio_embed = audio_embed[:, :, 2:]  # (B T E)
        phoneme_embed = self.phoneme_encoder(ph_seq)  # (B S E)
        # position embedding
        position = torch.arange(ph_seq.shape[1]).to(self.device)
        position = repeat(position, "S -> B S", B=ph_seq.shape[0])
        position = position / (ph_seq_lengths.unsqueeze(-1) + 1e-6)
        position = (
            torch.stack(
                (
                    position * (position < 1).float(),
                    (1 - position) * (position < 1).float(),
                ),
                -1,
            )
            + 1e-6
        )
        position_embedding = self.position_encoder(position)  # (B S E)
        phoneme_embed = phoneme_embed + position_embedding

        attn_logits = torch.matmul(audio_embed, phoneme_embed.transpose(1, 2)) / (
            phoneme_embed.shape[-1] ** 0.5
        )  # (B T S)
        ctc_logits = torch.cat(
            (
                ctc_blank_logits,
                attn_logits,
            ),
            dim=-1,
        )
        # attn_rms = (
        #     torch.linalg.vector_norm(attn_logits, ord=2, dim=-1)
        #     / (attn_logits.shape[-1] ** 0.5 + 1e-6)
        # ).unsqueeze(-1)
        attn_logits = torch.cat(
            (
                sp_logits,
                attn_logits,
            ),
            dim=-1,
        )
        return audio_embed, phoneme_embed, attn_logits, ctc_logits  # (B T S+1)

    def on_train_start(self):
        # resume loss schedulers
        for scheduler in self.losses_schedulers:
            scheduler.resume(self.global_step)
        self.losses_weights = self.losses_weights.to(self.device)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        (
            input_feature,  # (B C T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T), item in [0,S]
            label_type,  # (B)
        ) = batch

        audio_embed, phoneme_embed, attn_logits, ctc_logits = self.forward(
            input_feature.transpose(1, 2), input_feature_lengths, ph_seq, ph_seq_lengths
        )

        losses = self._get_loss(
            audio_embed,  # (B T E)
            phoneme_embed,  # (B S E)
            attn_logits,  # (B T S+1)
            ctc_logits,  # (B T S+1)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T) , item in [0,S]
            label_type,  # (B)
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
        self.log("train_loss/total", total_loss)
        return total_loss

    def on_validation_start(self):
        self.on_train_start()

    def validation_step(self, batch, batch_idx):
        (
            input_feature,  # (B C T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T), item in [0,S]
            label_type,  # (B)
        ) = batch

        ph_seq_g2p = ["SP"]
        for ph in ph_seq.squeeze(0).cpu().numpy():
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
        if ctc is not None:
            self.logger.experiment.add_text(
                f"valid/ctc_predict_{batch_idx}", " ".join(ctc), self.global_step
            )
        self.logger.experiment.add_figure(
            f"valid/plot_{batch_idx}", fig, self.global_step
        )

        audio_embed, phoneme_embed, attn_logits, ctc_logits = self.forward(
            input_feature.transpose(1, 2), input_feature_lengths, ph_seq, ph_seq_lengths
        )

        losses = self._get_loss(
            audio_embed,  # (B T E)
            phoneme_embed,  # (B S E)
            attn_logits,  # (B T S+1)
            ctc_logits,  # (B T S+1)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T)
            label_type,  # (B)
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

    def predict_step(self, batch, batch_idx):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx = batch
        if self.get_melspec is None:
            self.get_melspec = MelSpecExtractor(**self.melspec_config)

        waveform = load_wav(wav_path, self.device, self.melspec_config["sample_rate"])
        wav_length = waveform.shape[0] / self.melspec_config["sample_rate"]
        melspec = self.get_melspec(waveform).detach().unsqueeze(0)
        melspec = (melspec - melspec.mean()) / melspec.std()
        melspec = repeat(
            melspec, "B C T -> B C (T N)", N=self.melspec_config["scale_factor"]
        )
        (ph_seq, ph_intervals, word_seq, word_intervals, _, _) = self._infer_once(
            melspec, ph_seq, word_seq, ph_idx_to_word_idx, False, False
        )
        return wav_path, wav_length, ph_seq, ph_intervals, word_seq, word_intervals
