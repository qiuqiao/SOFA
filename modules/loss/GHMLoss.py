import torch
import torch.nn as nn


class GHMLoss(torch.nn.Module):
    def __init__(
            self,
            num_classes,
            num_loss_bins=10,
            alpha=0.99,
            label_smoothing=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classes_ema = torch.ones(num_classes)
        self.num_loss_bins = num_loss_bins
        self.loss_bins_ema = torch.ones(num_loss_bins)
        self.alpha = alpha
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

    def forward(self, pred, target, mask=None):
        # pred: [B, C, T]
        assert len(pred.shape) == 3 and pred.shape[1] == self.num_classes
        # target: [B, T] or [B, C, T]
        assert len(target.shape) == 2 or (len(target.shape) == 3 and target.shape[1] == self.num_classes)
        # mask: [B, T]
        if mask is not None:
            assert mask.shape[0] == target.shape[0] and mask.shape[1] == target.shape[-1]

        raw_loss = self.loss_fn(pred, target)  # [B, T]

        if pred.shape != target.shape:
            target_probs = (torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1))  # [B, C, T]
        else:
            target_probs = target  # [B, C, T]
        pred_probs = torch.softmax(pred, dim=1)  # [B, C, T]

        # calculate weighted loss
        # 这里与原版实现不同，这里用了L1 loss，而不是原版的“真实标签的概率”。这是为了兼容概率张量输入。
        # 这里的L1_loss值域是[0,2]，除以2以保证范围在[0,1]
        L1_loss = (pred_probs - target_probs).abs().sum(dim=1).clamp(1e-6, 2 - 1e-6) / 2  # [B, T]
        loss_weighted = raw_loss / torch.sqrt(
            (self.classes_ema.unsqueeze(0).unsqueeze(-1) * torch.sqrt(pred_probs * target_probs)).sum(dim=1)
            * self.loss_bins_ema[torch.floor(L1_loss * self.num_loss_bins).long()]
            + 1e-10
        )  # [B, T]

        # apply mask
        if mask is not None:
            loss_weighted = loss_weighted * mask
            loss_final = torch.sum(loss_weighted) / torch.sum(mask)
        else:
            loss_final = torch.mean(loss_weighted)

        # update ema
        # "Elements lower than min and higher than max and NaN elements are ignored."
        if mask is not None:
            L1_loss = L1_loss + 10 * (mask - 1)
            classes_stat = (torch.sqrt(pred_probs * target_probs) * mask.unsqueeze(1)).sum(dim=0).sum(dim=-1)
        else:
            classes_stat = (torch.sqrt(pred_probs * target_probs)).sum(dim=0).sum(dim=-1)
        loss_bins_stat = torch.histc(L1_loss, bins=self.num_loss_bins, min=0, max=1)
        self.loss_bins_ema = self.update_ema(self.loss_bins_ema, self.num_loss_bins, loss_bins_stat)
        self.classes_ema = self.update_ema(self.classes_ema, self.num_classes, classes_stat)

        return loss_final

    def update_ema(self, ema, num_bins, stat):
        stat = stat / (torch.sum(stat) + 1e-10) * num_bins
        ema = ema * self.alpha + (1 - self.alpha) * stat
        ema = ema / (torch.sum(ema) + 1e-10) * num_bins
        return ema
