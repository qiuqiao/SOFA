import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler_module
import yaml
from einops import rearrange, repeat

from modules.layer.model.forced_alignment import ForcedAlignmentModel
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
        self.melspec_config = melspec_config
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        self.data_augmentation_enabled = data_augmentation_enabled
        # model
        self.model = ForcedAlignmentModel(
            n_mels=melspec_config["n_mels"],
            vocab_size=self.vocab["<vocab_size>"],
            **model_config,
        )

        # get_melspec
        self.get_melspec = None

        # validation_step_outputs
        self.validation_step_outputs = None

    def load_pretrained(self, pretrained_model):
        raise NotImplementedError

    def set_inference_mode(self, mode):
        raise NotImplementedError

    def _losses_schedulers_step(self):
        raise NotImplementedError
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        raise NotImplementedError
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(
            self.device
        )

    def _decode(self, ph_seq_id, alignment_matrix):
        # ph_seq_id: (S)
        # alignment_matrix: (T, S)
        raise NotImplementedError
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
        raise NotImplementedError
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
        attn_logits,  # (B T S)
        attn_target,  # (B T), item in [0,S]
        input_feature_lengths,  # (B)
        ph_seq_lengths,  # (B)
        valid,
    ):
        # 先写ctc的loss
        raise NotImplementedError
        B, T, S = attn_logits.shape
        # 如果add了prior就不需要mask，prior里带了
        # (B T)
        feature_len_mask = (
            (
                torch.arange(T).to(self.device).unsqueeze(0)
                < (input_feature_lengths.unsqueeze(1))
            )
            .to(self.device)
            .detach()
        )
        # (B S)
        sequence_len_mask = (
            (
                torch.arange(S).to(self.device).unsqueeze(0)
                < (ph_seq_lengths.unsqueeze(1))
            )
            .to(self.device)
            .detach()
        )
        # (B T S)
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
        self, attn_log_prob, input_feature_lengths, ph_seq_lengths, valid
    ):
        B, T, S = attn_log_prob.shape

        attn_log_prob
        attn_log_prob = rearrange(attn_log_prob, "B T S -> T B S")
        blank_log_prob = (
            torch.ones(T, B, 1).to(self.device) * self.loss_config["blank_log_prob"]
        ).detach()
        attn_log_prob = torch.cat([blank_log_prob, attn_log_prob], dim=-1)

        targets = torch.arange(max(ph_seq_lengths)).to(self.device) + 1
        targets = repeat(targets, "S -> B S", B=B)
        targets = targets * (targets < (ph_seq_lengths.unsqueeze(1) + 1)).float()
        targets = targets.long()

        # ctc loss
        # TODO: use ghm loss
        loss = nn.functional.ctc_loss(
            attn_log_prob,
            targets,
            input_lengths=input_feature_lengths,
            target_lengths=ph_seq_lengths,
            # reduction="sum",
        )

        return loss

    def _get_consistency_loss(
        self, audio_embed, input_feature_lengths, ph_seq_lengths, valid
    ):
        raise NotImplementedError
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
        attn_log_prob,  # (B T S)
        input_feature_lengths,  # (B)
        ph_seq_lengths,  # (B)
        attn_target,  # (B T) , item in [0,S]
        label_type,  # (B)
        valid=False,
    ):
        full_label_idx = label_type >= 2
        weak_label_idx = label_type >= 1
        ZERO = torch.tensor(0).to(self.device).detach()

        if (full_label_idx).any():
            # full_label_loss = self._get_full_label_loss(
            #     attn_logits[full_label_idx],
            #     attn_target[full_label_idx],
            #     input_feature_lengths[full_label_idx],
            #     ph_seq_lengths[full_label_idx],
            #     valid,
            # )
            frame_classification_loss = ZERO
        else:
            frame_classification_loss = ZERO

        if (weak_label_idx).any():
            forward_sum_loss = self._get_weak_label_loss(
                attn_log_prob[weak_label_idx],
                input_feature_lengths[weak_label_idx],
                ph_seq_lengths[weak_label_idx],
                valid,
            )
        else:
            forward_sum_loss = ZERO

        if not valid and self.data_augmentation_enabled:
            # consistency_loss = self._get_consistency_loss(
            #     audio_embed, input_feature_lengths, ph_seq_lengths, valid
            # )
            # # TODO: add reconstruction loss
            consistency_loss = ZERO
        else:
            consistency_loss = ZERO

        # TODO: add weight and scheduler
        total_loss = frame_classification_loss + forward_sum_loss + consistency_loss
        # TODO: add scheduler
        losses_dict = {
            "total_loss": total_loss,
            "frame_classification_loss": frame_classification_loss,
            "forward_sum_loss": forward_sum_loss,
            "consistency_loss": consistency_loss,
        }

        return total_loss, losses_dict

    def configure_optimizers(self):
        optimizer_params = []
        scheduler_learning_rates = []
        for k, v in self.optimizer_config["lr"].items():
            optimizer_params.append(
                {
                    "params": getattr(self.model, k).parameters(),
                    "lr": v,
                }
            )
            scheduler_learning_rates.append(v)

        optimizer = torch.optim.AdamW(
            optimizer_params,
            weight_decay=self.optimizer_config["weight_decay"],
        )
        scheduler = {
            "scheduler": lr_scheduler_module.OneCycleLR(
                optimizer,
                max_lr=scheduler_learning_rates,
                total_steps=self.optimizer_config["total_steps"],
            ),
            "interval": "step",
        }

        for k, v in self.optimizer_config["freeze"].items():
            if v:
                getattr(self.model, k).requires_grad_(False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, input_feature, ph_seq):
        attn_logits = self.model(input_feature, ph_seq)
        attn_log_prob = nn.functional.log_softmax(attn_logits, dim=-1)
        return attn_log_prob

    def on_train_start(self):
        pass
        # resume loss schedulers
        # for scheduler in self.losses_schedulers:
        #     scheduler.resume(self.global_step)
        # self.losses_weights = self.losses_weights.to(self.device)

    def training_step(self, batch, batch_idx):
        (
            input_feature,  # (B C T)
            input_feature_lengths,  # (B)
            ph_seq,  # (B S)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T), item in [0,S]
            label_type,  # (B)
        ) = batch

        attn_log_prob = self.forward(input_feature.transpose(1, 2), ph_seq)

        total_loss, losses_dict = self._get_loss(
            attn_log_prob,  # (B T S)
            input_feature_lengths,  # (B)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T) , item in [0,S]
            label_type,  # (B)
            valid=False,
        )

        # TODO: scheduler step

        log_dict = {"train/" + k: v for k, v in losses_dict.items() if v != 0}
        log_dict["scheduler/lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        # TODO: log scheduler

        self.log_dict(log_dict)
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

        attn_log_prob = self.forward(input_feature.transpose(1, 2), ph_seq)

        total_loss, losses_dict = self._get_loss(
            attn_log_prob,  # (B T S)
            input_feature_lengths,  # (B)
            ph_seq_lengths,  # (B)
            attn_target,  # (B T) , item in [0,S]
            label_type,  # (B)
            valid=False,
        )

        log_dict = {"valid/" + k: v for k, v in losses_dict.items() if v != 0}
        fig = plot_for_valid(
            melspec=input_feature.cpu().numpy(),
            ph_seq=None,
            ph_intervals=None,
            frame_confidence=None,
            alignment_spec=attn_log_prob.exp().cpu().numpy(),
            ph_frame_id_gt=None,
        )
        self.logger.experiment.add_figure(f"plot_{batch_idx}", fig, self.global_step)
        if self.validation_step_outputs is None:
            self.validation_step_outputs = [log_dict]
        else:
            self.validation_step_outputs.append(log_dict)

    def on_validation_epoch_end(self):
        res_log_dict = {}
        for log_dict in self.validation_step_outputs:
            for k, v in log_dict.items():
                if k not in res_log_dict:
                    res_log_dict[k] = []
                res_log_dict[k].append(v)

        for k, v in res_log_dict.items():
            res_log_dict[k] = torch.stack(v).mean() if len(v) > 0 else 0

        self.log_dict({k: v for k, v in res_log_dict.items() if v != 0})

        self.validation_step_outputs = None

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError
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
