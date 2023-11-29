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
        # 这里与原版实现不同，这里用了L1 loss，而不是原版的梯度模长。这是为了兼容概率张量输入。
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
        loss_weighted = raw_loss / weight  # [B, T]

        # apply mask
        if mask is not None:
            loss_weighted = loss_weighted * mask.float()
            loss_final = torch.sum(loss_weighted) / torch.sum(mask.float())
        else:
            loss_final = torch.mean(loss_weighted)

        if not valid:
            # update ema
            # "Elements lower than min and higher than max and NaN elements are ignored."
            if mask is not None:
                L1_loss = L1_loss - 10 * mask.logical_not()
                classes_hist = (
                    (
                        torch.sqrt(pred_probs * target_probs)
                        * (mask.float()).unsqueeze(1)
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


class BCEGHMLoss(torch.nn.Module):
    def __init__(self, num_bins=10, alpha=1 - 1e-6, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction="none")
        self.num_bins = num_bins
        self.register_buffer("GD_stat_ema", torch.ones(num_bins))
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, pred_porb, target_porb, mask=None, valid=False):
        if len(pred_porb) <= 0:
            return torch.tensor(0.0).to(pred_porb.device)
        if mask is None:
            mask = torch.ones_like(pred_porb).to(pred_porb.device)
        assert (
            pred_porb.shape == target_porb.shape
            and pred_porb.shape[:2] == mask.shape[:2]
        )
        if len(mask.shape) < len(pred_porb.shape):
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, pred_porb.shape[-1])
        assert pred_porb.max() <= 1 and pred_porb.min() >= 0
        assert target_porb.max() <= 1 and target_porb.min() >= 0

        target_porb = target_porb.clamp(self.label_smoothing, 1 - self.label_smoothing)

        raw_loss = self.loss_fn(pred_porb, target_porb)

        gradient_magnitudes = (pred_porb - target_porb).abs()
        gradient_magnitudes_index = (
            torch.floor(gradient_magnitudes * self.num_bins).long().clamp(0, 9)
        )
        weights = 1 / self.GD_stat_ema[gradient_magnitudes_index] + 1e-3
        loss_weighted = raw_loss * weights
        mask_weights = mask.float()
        loss_weighted = loss_weighted * mask_weights
        loss_final = torch.sum(loss_weighted) / torch.sum(mask_weights)

        if not valid:
            # update ema
            # "Elements lower than min and higher than max and NaN elements are ignored."
            gradient_magnitudes_index = gradient_magnitudes_index.flatten()
            mask_weights = mask_weights.flatten()
            gradient_magnitudes_index_hist = torch.bincount(
                input=gradient_magnitudes_index,
                weights=mask_weights,
                minlength=self.num_bins,
            )
            self.GD_stat_ema = update_ema(
                self.GD_stat_ema,
                self.alpha,
                self.num_bins,
                gradient_magnitudes_index_hist,
            )

        return loss_final


class MultiLabelGHMLoss(torch.nn.Module):
    def __init__(self, num_classes, num_bins=10, alpha=(1 - 1e-6), label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.num_bins = num_bins
        # 难易样本不均衡
        self.register_buffer("GD_stat_ema", torch.ones(num_bins))
        self.num_classes = num_classes
        # 类别不均衡与正负样本不均衡，分为正、负、中性三类
        self.register_buffer("label_stat_ema_each_class", torch.ones([num_classes * 3]))
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, pred_logits, target_porb, mask=None, valid=False):
        """_summary_

        Args:
            pred_porb (torch.Tensor): predicted probability, shape: (** C)
            target_porb (torch.Tensor): target probability, shape: same as pred_porb.
            mask (torch.Tensor, optional): mask tensor, ignore loss when mask==0.
                                           shape: same as pred_porb. Defaults to None.
            valid (bool, optional): enable ema update. Defaults to False.

        Returns:
            loss_final (torch.Tensor): loss value, shape: ()
        """
        if len(pred_logits) <= 0:
            return torch.tensor(0.0).to(pred_logits.device)
        if mask is None:
            mask = torch.ones_like(pred_logits).to(pred_logits.device)
        assert (
            pred_logits.shape == target_porb.shape
            and pred_logits.shape[:2] == mask.shape[:2]
        )
        if len(mask.shape) < len(pred_logits.shape):
            mask = mask.unsqueeze(-1)
        assert pred_logits.shape[-1] == self.num_classes
        assert target_porb.max() <= 1 and target_porb.min() >= 0

        pred_logits = pred_logits.reshape(-1, self.num_classes)
        target_porb = target_porb.reshape(-1, self.num_classes)
        mask = mask.reshape(target_porb.shape[0], -1)
        if mask.shape[-1] == 1 and target_porb.shape[-1] > 1:
            mask = mask.repeat(1, target_porb.shape[-1])
        target_porb = target_porb.clamp(self.label_smoothing, 1 - self.label_smoothing)

        raw_loss = self.loss_fn(pred_logits, target_porb)

        pred_porb = torch.nn.functional.sigmoid(pred_logits)
        gradient_magnitudes_index = (
            torch.floor((pred_porb - target_porb).abs() * self.num_bins)
            .long()
            .clamp(0, self.num_bins - 1)
        )
        GD_weights = 1 / self.GD_stat_ema[gradient_magnitudes_index] + 1e-3
        target_porb_index = torch.floor(target_porb * 3).long().clamp(
            0, 2
        ) + 3 * torch.arange(self.num_classes).to(target_porb.device).unsqueeze(0)
        classes_weights = 1 / self.label_stat_ema_each_class[target_porb_index] + 1e-3
        weights = torch.sqrt(GD_weights * classes_weights)
        loss_weighted = raw_loss * weights
        loss_weighted = loss_weighted * mask
        loss_final = torch.sum(loss_weighted) / torch.sum(mask)

        if not valid:
            # update ema
            # TODO:要带着mask统计的话，mask的shape就要和input一致
            mask = mask.flatten()
            gradient_magnitudes_index = gradient_magnitudes_index.flatten()
            gradient_magnitudes_index_hist = torch.bincount(
                input=gradient_magnitudes_index,
                weights=mask,
                minlength=self.num_bins,
            )
            self.GD_stat_ema = update_ema(
                self.GD_stat_ema,
                self.alpha,
                self.num_bins,
                gradient_magnitudes_index_hist,
            )

            target_porb_index = target_porb_index.flatten()
            target_porb_index_hist = torch.bincount(
                input=target_porb_index,
                weights=mask,
                minlength=self.num_classes * 3,
            )
            self.label_stat_ema_each_class = update_ema(
                self.label_stat_ema_each_class,
                self.alpha,
                self.num_classes * 3,
                target_porb_index_hist,
            )

        return loss_final


if __name__ == "__main__":
    loss_fn = MultiLabelGHMLoss(10, alpha=0.9)
    input = torch.nn.functional.sigmoid(torch.randn(3, 3, 10) * 10)
    target = (torch.nn.functional.sigmoid(torch.randn(3, 3, 10)) > 0.5).float()
    print(loss_fn(input, target))
