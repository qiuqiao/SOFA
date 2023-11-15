import torch
import torch.nn as nn


def update_ema(ema, alpha, num_bins, hist):
    hist = hist / (torch.sum(hist) + 1e-10) * num_bins
    ema = ema * alpha + (1 - alpha) * hist
    ema = ema / (torch.sum(ema) + 1e-10) * num_bins
    return ema


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
        self.register_buffer("classes_ema", torch.ones(num_classes))
        self.num_loss_bins = num_loss_bins
        self.register_buffer("loss_bins_ema", torch.ones(num_loss_bins))
        self.alpha = alpha
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

    def forward(self, pred, target, mask=None, valid=False):
        if len(pred) <= 0:
            return torch.tensor(0.0).to(pred.device)
        # pred: [B, C, T]
        assert len(pred.shape) == 3
        assert pred.shape[1] == self.num_classes
        # target: [B, T] or [B, C, T]
        if len(target.shape) == 2:
            target = target.long()
        assert len(target.shape) == 2 or (
            len(target.shape) == 3 and target.shape[1] == self.num_classes
        )
        # mask: [B, T]
        if mask is not None:
            assert (
                mask.shape[0] == target.shape[0] and mask.shape[1] == target.shape[-1]
            )

        raw_loss = self.loss_fn(pred, target)  # [B, T]

        if pred.shape != target.shape:
            target_probs = (
                torch.zeros_like(pred)
                .to(pred.device)
                .scatter_(1, target.unsqueeze(1).to(torch.int64), 1)
            )  # [B, C, T]
        else:
            target_probs = target  # [B, C, T]
        pred_probs = torch.softmax(pred, dim=1).detach()  # [B, C, T]

        # calculate weighted loss
        # 这里与原版实现不同，这里用了L1 loss，而不是原版的“真实标签的概率”。这是为了兼容概率张量输入。
        # 这里的L1_loss值域是[0,2]，除以2以保证范围在[0,1]
        L1_loss = (pred_probs - target_probs).abs().sum(dim=1).clamp(
            1e-6, 2 - 1e-6
        ) / 2  # [B, T]
        weight = torch.sqrt(
            (
                self.classes_ema.unsqueeze(0).unsqueeze(-1)
                * torch.sqrt(pred_probs * target_probs)
            ).sum(dim=1)
            * self.loss_bins_ema[torch.floor(L1_loss * self.num_loss_bins).long()]
            + 1e-10
        )  # [B, T]
        # print(weight.mean(), weight.shape)
        loss_weighted = raw_loss / weight  # [B, T]

        # apply mask
        if mask is not None:
            loss_weighted = loss_weighted * mask.logical_not().float()
            loss_final = torch.sum(loss_weighted) / torch.sum(
                mask.logical_not().float()
            )
        else:
            loss_final = torch.mean(loss_weighted)

        if not valid:
            # update ema
            # "Elements lower than min and higher than max and NaN elements are ignored."
            if mask is not None:
                L1_loss = L1_loss - 10 * mask
                classes_hist = (
                    (
                        torch.sqrt(pred_probs * target_probs)
                        * (mask.logical_not().float()).unsqueeze(1)
                    )
                    .sum(dim=0)
                    .sum(dim=-1)
                )  # [C]
            else:
                classes_hist = (
                    (torch.sqrt(pred_probs * target_probs)).sum(dim=0).sum(dim=-1)
                )  # [C]
            loss_hist = torch.histc(L1_loss, bins=self.num_loss_bins, min=0, max=1)
            self.loss_bins_ema = update_ema(
                self.loss_bins_ema, self.alpha, self.num_loss_bins, loss_hist
            )
            self.classes_ema = update_ema(
                self.classes_ema, self.alpha, self.num_classes, classes_hist
            )

        return loss_final


class CTCGHMLoss(torch.nn.Module):
    def __init__(self, num_bins=10, alpha=0.999):
        super().__init__()
        self.ctc_loss_fn = nn.CTCLoss(reduction="none")
        self.ctc_loss_fn_cpu = nn.CTCLoss(reduction="none").cpu()
        self.num_bins = num_bins
        self.register_buffer("ema", torch.ones(num_bins))
        self.alpha = alpha

    def forward(self, log_probs, targets, input_lengths, target_lengths, valid=False):
        if len(log_probs) <= 0:
            return torch.tensor(0.0).to(log_probs.device)
        try:
            raw_loss = self.ctc_loss_fn(
                log_probs, targets, input_lengths, target_lengths
            )
        except RuntimeError:
            raw_loss = self.ctc_loss_fn_cpu(
                log_probs.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu(),
            ).to(log_probs.device)
        loss_for_ema = (
            (-raw_loss / input_lengths).exp().clamp(1e-6, 1 - 1e-6)
        )  # 值域为[0, 1]
        loss_weighted = raw_loss / (
            self.ema[torch.floor(loss_for_ema * self.num_bins).long()] + 1e-10
        )
        loss_final = loss_weighted.mean()

        if not valid:
            hist = torch.histc(loss_for_ema, bins=self.num_bins, min=0, max=1)
            self.ema = update_ema(self.ema, self.alpha, self.num_bins, hist)

        return loss_final
