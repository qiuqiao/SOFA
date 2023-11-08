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
import numpy as np
from modules.utils.load_wav import load_wav
from modules.utils.plot import plot_for_test


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
        if not self.infer_params["phoneme"]:
            with open(kwargs["dictionary"], 'r') as f:
                dictionary = f.read().strip().split('\n')
            self.infer_params["dictionary"] = {item.split('\t')[0].strip(): item.split('\t')[1].strip().split(' ')
                                               for item in dictionary}

        self.infer_params["get_melspec"] = MelSpecExtractor(**self.hparams.melspec_config,
                                                            device=self.device)

    @staticmethod
    def _decode(ph_seq_id, ph_prob_log, edge_prob):
        # ph_seq_id: (T)
        # ph_prob_log: (T, vocab_size)
        # edge_prob: (T,2)
        T = ph_prob_log.shape[0]
        S = len(ph_seq_id)

        edge_prob_log = np.log(edge_prob).astype("float64")
        not_edge_prob_log = np.log(1 - edge_prob).astype("float64")
        # 乘上is_phoneme正确分类的概率 TODO: enable this
        # ph_prob_log[:, 0] += ph_prob_log[:, 0]
        # ph_prob_log[:, 1:] += 1 / ph_prob_log[:, [0]]

        # init
        dp = np.zeros([T, S]).astype("float64") - np.inf  # (T, S)
        backtrack_s = np.zeros_like(dp).astype("int32") - 1
        # 只能从SP开始或者从第一个音素开始
        dp[0, 0] = ph_prob_log[0, 0]
        dp[0, 1] = ph_prob_log[0, ph_seq_id[1]]
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
        ph_seq_id_pred = []
        ph_time_int = []
        frame_confidence = []
        # 只能从最后一个音素或者SP结束
        if dp[-1, -2] > dp[-1, -1]:
            s = S - 2
        else:
            s = S - 1
        for t in np.arange(T - 1, -1, -1):
            assert backtrack_s[t, s] >= 0 or t == 0
            frame_confidence.append(dp[t, s])
            if backtrack_s[t, s] != 0:
                ph_seq_id_pred.append(ph_seq_id[s])
                ph_time_int.append(t)
                s -= backtrack_s[t, s]
        ph_seq_id_pred.reverse()
        ph_time_int.reverse()
        frame_confidence.reverse()
        frame_confidence = np.exp(np.diff(np.pad(frame_confidence, (1, 0), 'constant', constant_values=0.), 1))

        return (
            np.array(ph_seq_id_pred),
            np.array(ph_time_int),
            np.array(frame_confidence),
        )

    def _infer_once(self, wav_path, return_ctc=False, return_plot=False):

        lab_path = wav_path.parent / f"{wav_path.stem}.lab"
        if not lab_path.exists():
            return None, None, None, None

        # TODO: add matching mode
        with open(lab_path, 'r') as f:
            word_seq = f.read().strip().split(' ')
        ph_seq = [0]
        ph_word_idx = [-1]
        if not self.infer_params["phoneme"]:
            for idx, word in enumerate(word_seq):
                phones = self.infer_params["dictionary"][word]
                ph_seq.extend(phones)
                ph_word_idx.extend([idx] * len(phones))
                ph_seq.append(0)
                ph_word_idx.append(-1)
        else:
            for ph in word_seq:
                ph_seq.append(ph)
                ph_seq.append(0)
        ph_seq_id = np.array([self.vocab[ph] if ph != 0 else 0 for ph in ph_seq])

        # forward
        waveform = load_wav(wav_path, self.device, self.melspec_config["sample_rate"])
        melspec = self.infer_params["get_melspec"](waveform).unsqueeze(0)
        melspec = (melspec - melspec.mean()) / melspec.std()
        with torch.no_grad():
            (
                ph_frame_pred,  # (B, T, vocab_size)
                ph_edge_pred,  # (B, T, 2)
                ctc_pred,  # (B, T, vocab_size)
            ) = self.forward(melspec.transpose(1, 2))

        ph_frame_pred = ph_frame_pred.squeeze(0)
        ph_edge_pred = ph_edge_pred.squeeze(0)
        ctc_pred = ctc_pred.squeeze(0)

        # decode
        edge = torch.softmax(ph_edge_pred, dim=-1).cpu().numpy().astype("float64")
        edge_diff = np.pad(np.diff(edge[:, 0]), (1, 0), "constant", constant_values=0)
        edge_prob = np.pad(edge[1:, 0] + edge[:-1, 0], (1, 0), "constant", constant_values=edge[0, 0] * 2).clip(0, 1)

        ph_prob_log = torch.log_softmax(ph_frame_pred, dim=-1).cpu().numpy().astype("float64")
        (
            ph_seq_id_pred,
            ph_time_int_pred,
            frame_confidence,
        ) = self._decode(
            ph_seq_id,
            ph_prob_log,
            edge_prob,
        )

        # postprocess
        ph_seq_pred = np.array([self.vocab[ph] for ph in ph_seq_id_pred])
        ph_time_fractional = (edge_diff[ph_time_int_pred] / 2).clip(-0.5, 0.5)
        ph_time_pred = ph_time_int_pred.astype("float64") + ph_time_fractional
        ph_time_pred = np.concatenate([ph_time_pred, [ph_frame_pred.shape[0]]])
        ph_time_pred = ph_time_pred * (self.melspec_config["hop_length"] / self.melspec_config["sample_rate"])
        ph_time_interval = np.stack([ph_time_pred[:-1], ph_time_pred[1:]], axis=1)

        # ctc decode
        ctc = None
        if return_ctc:
            ctc = torch.argmax(ctc_pred, dim=-1).cpu().numpy()
            ctc_index = np.concatenate([[0], ctc])
            ctc_index = (ctc_index[1:] != ctc_index[:-1]) * ctc != 0
            ctc = ctc[ctc_index]
            ctc = np.array([self.vocab[ph] for ph in ctc if ph != 0])

        # plot
        fig = None
        if return_plot:
            ph_frame_idx = np.zeros(ph_frame_pred.shape[0], dtype="int32")
            ph_frame_idx[ph_time_int_pred] = 1
            ph_frame_idx = ph_frame_idx.cumsum() - 1
            ph_frame_id_gt = ph_seq_id_pred[ph_frame_idx]
            args = {
                "melspec": melspec.cpu().numpy(),
                "ph_seq": ph_seq_pred,
                "ph_time": ph_time_int_pred.astype("float64") + ph_time_fractional,
                "frame_confidence": frame_confidence,

                "ph_frame_prob": torch.softmax(ph_frame_pred, dim=-1).cpu().numpy(),
                "ph_frame_id_gt": ph_frame_id_gt,
                "edge_prob": edge_prob,
            }
            fig = plot_for_test(**args)

        return ph_seq_pred, ph_time_interval, ctc, fig

    def predict_step(self, batch, batch_idx):
        ph_seq, ph_time_interval, _, _ = self._infer_once(batch, False, False)
        return ph_seq, ph_time_interval

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
