import torch
from einops import rearrange, repeat
from torch import nn

import modules.scheduler as scheduler_module
from modules.loss.binary_emd_loss import BinaryEMDLoss
from modules.loss.ghm_Loss import CTCGHMLoss, GHMLoss, MultiLabelGHMLoss


class ForcedAlignmentLoss(nn.Module):

    def __init__(self, loss_config, total_steps, vocab_size, data_augmentation_enabled):
        super(ForcedAlignmentLoss, self).__init__()
        self.device = None
        self.data_augmentation_enabled = data_augmentation_enabled
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
        self.losses_weights = torch.tensor(loss_config["losses"]["weights"])

        self.losses_schedulers = []
        for enabled in loss_config["losses"]["enable_RampUpScheduler"]:
            if enabled:
                self.losses_schedulers.append(
                    scheduler_module.GaussianRampUpScheduler(max_steps=total_steps)
                )
            else:
                self.losses_schedulers.append(scheduler_module.NoneScheduler())

        self.ph_frame_GHM_loss_fn = GHMLoss(
            vocab_size,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            loss_config["function"]["label_smoothing"],
        )
        self.pseudo_label_GHM_loss_fn = MultiLabelGHMLoss(
            vocab_size,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            loss_config["function"]["label_smoothing"],
        )
        self.ph_edge_GHM_loss_fn = MultiLabelGHMLoss(
            1,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.EMD_loss_fn = BinaryEMDLoss()
        self.ph_edge_diff_GHM_loss_fn = MultiLabelGHMLoss(
            1,
            loss_config["function"]["num_bins"],
            loss_config["function"]["alpha"],
            label_smoothing=0.0,
        )
        self.MSE_loss_fn = nn.MSELoss()
        self.CTC_GHM_loss_fn = CTCGHMLoss(alpha=1 - 1e-3)

    def _losses_schedulers_step(self):
        for scheduler in self.losses_schedulers:
            scheduler.step()

    def _losses_schedulers_call(self):
        return torch.tensor([scheduler() for scheduler in self.losses_schedulers]).to(
            self.device
        )

    def _get_loss(
        self,
        ph_frame_logits,  # (B, T, vocab_size)
        ph_edge_logits,  # (B, T)
        ctc_logits,  # (B, T, vocab_size)
        ph_frame_gt,  # (B, T)
        ph_edge_gt,  # (B, T)
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
        ZERO = torch.tensor(0).to(self.device)

        if (full_label_idx).any():
            (
                ph_frame_GHM_loss,
                ph_edge_GHM_loss,
                ph_edge_EMD_loss,
                ph_edge_diff_loss,
            ) = self._get_full_label_loss(
                ph_frame_logits[full_label_idx, :, :],
                ph_edge_logits[full_label_idx, :],
                ph_frame_gt[full_label_idx, :],
                ph_edge_gt[full_label_idx, :],
                input_feature_lengths[full_label_idx],
                ph_mask[full_label_idx, :],
                valid,
            )
        else:
            ph_frame_GHM_loss = ph_edge_GHM_loss = ZERO
            ph_edge_EMD_loss = ph_edge_diff_loss = ZERO

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
                ph_frame_logits, ph_edge_logits, input_feature_lengths
            )
            pseudo_label_loss = ZERO
            # pseudo_label_loss = self._get_pseudo_label_loss(
            #     ph_frame_logits[not_full_label_idx, :, :],
            #     input_feature_lengths[not_full_label_idx],
            #     valid,
            # )
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

        scheduler_weights = self._losses_schedulers_call()
        total_loss = (
            torch.stack(losses) * self.losses_weights * scheduler_weights
        ).sum()
        losses.append(total_loss)

        losses_dict = {
            f"loss/{k}": v for k, v in zip(self.losses_names, losses) if v != 0
        }

        schedulers_dict = {
            f"scheduler/{k}": v
            for k, v in zip(self.losses_names, scheduler_weights)
            if v != 1
        }

        return total_loss, losses_dict, schedulers_dict

    def _get_full_label_loss(
        self,
        ph_frame_logits,
        ph_edge_logits,
        ph_frame_gt,
        ph_edge_gt,
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
        # print((mask.unsqueeze(-1) * ph_mask.unsqueeze(1)).shape, ph_frame_pred.shape)
        ph_frame_GHM_loss = self.ph_frame_GHM_loss_fn(
            ph_frame_logits,
            ph_frame_gt,
            (mask.unsqueeze(-1) * ph_mask.unsqueeze(1)),
            valid,
        )

        # ph_edge loss
        # BCE_GHM loss
        ph_edge_GHM_loss = self.ph_edge_GHM_loss_fn(
            ph_edge_logits.unsqueeze(-1), ph_edge_gt.unsqueeze(-1), mask, valid
        )

        # EMD loss
        ph_edge_pred = torch.nn.functional.sigmoid(ph_edge_logits.float())
        ph_edge_EMD_loss = self.EMD_loss_fn(ph_edge_pred * mask, ph_edge_gt * mask)

        # diff loss
        ph_edge_diff_loss = self.ph_edge_diff_GHM_loss_fn(
            (torch.diff(ph_edge_logits, 1, dim=-1) + 1).unsqueeze(-1) / 2,
            (torch.diff(ph_edge_gt, 1, dim=-1) + 1).unsqueeze(-1) / 2,
            mask[:, 1:],
            valid,
        )
        return ph_frame_GHM_loss, ph_edge_GHM_loss, ph_edge_EMD_loss, ph_edge_diff_loss

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

    def _get_consistency_loss(
        self, ph_frame_logits, ph_edge_logits, input_feature_lengths
    ):
        output_tensors = torch.cat(
            [ph_frame_logits, ph_edge_logits.unsqueeze(-1)], dim=-1
        )
        output_tensors = torch.nn.functional.sigmoid(output_tensors.float())
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

    def _get_pseudo_label_loss(self, ph_frame_logits, input_feature_lengths, valid):
        B = ph_frame_logits.shape[0]
        T = ph_frame_logits.shape[1]

        ph_edge_prob = torch.nn.functional.sigmoid(ph_frame_logits.float())

        pred1 = ph_edge_prob[: B // 2, :]
        pred2 = ph_edge_prob[B // 2 :, :]
        pseudo_label1 = (pred1 >= 0.5).float()
        pseudo_label2 = (pred2 >= 0.5).float()
        gradient_magnitude1 = torch.abs(pred1 - pseudo_label1)
        gradient_magnitude2 = torch.abs(pred2 - pseudo_label2)
        gradient_magnitude = (gradient_magnitude1 + gradient_magnitude2) / 2

        # calculate mask matrix
        # (B//2, T, 1)
        mask = torch.arange(T).to(self.device)
        mask = repeat(mask, "T -> B T", B=B // 2)
        mask = (
            (mask < input_feature_lengths[: B // 2].unsqueeze(1))
            .to(torch.bool)
            .unsqueeze(-1)
        )
        pseudo_label_mask = (  # (B//2, T)
            mask
            & (pseudo_label1 == pseudo_label2)
            & (gradient_magnitude < self.pseudo_label_auto_theshold)
        )

        if pseudo_label_mask.sum() / mask.sum() < self.pseudo_label_ratio:
            self.pseudo_label_auto_theshold += 0.005
        else:
            self.pseudo_label_auto_theshold -= 0.005

        if pseudo_label_mask.any():
            pseudo_label_loss = self.pseudo_label_GHM_loss_fn(
                ph_frame_logits,
                torch.cat([pseudo_label1, pseudo_label2], dim=0),
                torch.cat([pseudo_label_mask, pseudo_label_mask], dim=0),
                valid,
            )
        else:
            pseudo_label_loss = torch.tensor(0).to(self.device)

        return pseudo_label_loss

    def resume(self, global_step):
        for scheduler in self.losses_schedulers:
            scheduler.resume(global_step)

    def to(self, device):
        self.device = device
        self.losses_weights = self.losses_weights.to(device)
        super(ForcedAlignmentLoss, self).to(device)
        return self
