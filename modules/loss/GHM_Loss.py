import torch
import torch.nn as nn


class GHMLoss(torch.nn.Module):
    def __init__(
        self,
        device,
        num_classes,
        num_prob_bins=10,
        alpha=0.99,
        label_smoothing=0.0,
        enable_prob_input=False,
    ):
        super().__init__()
        self.device = device
        self.enable_prob_input = enable_prob_input
        self.num_classes = num_classes
        self.num_prob_bins = num_prob_bins
        if not enable_prob_input:
            self.classes_ema = torch.ones(num_classes).to(self.device)
        self.prob_bins_ema = torch.ones(num_prob_bins).to(self.device)
        self.alpha = alpha
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

    def forward(self, pred, target):
        pred_prob = torch.softmax(pred, dim=1)
        if not self.enable_prob_input:
            target_prob = (
                torch.zeros_like(pred_prob)
                .scatter_(1, target.unsqueeze(1), 1)
                .to(self.device)
            )
        else:
            target_prob = target
        pred_prob = (pred_prob * target_prob).sum(dim=1).clamp(1e-6, 1 - 1e-6)

        loss = self.loss_fn(pred, target)
        if not self.enable_prob_input:
            loss_classes = target.long()
            # print(len(self.classes_ema),loss_classes.min().cpu().numpy(),loss_classes.max().cpu().numpy(),len(self.prob_bins_ema),torch.floor(pred_prob*self.num_prob_bins).long().min().cpu().numpy(),torch.floor(pred_prob*self.num_prob_bins-1e-10).long().max().cpu().numpy())
            loss_weighted = loss / torch.sqrt(
                (
                    self.classes_ema[loss_classes]
                    * self.prob_bins_ema[
                        torch.floor(pred_prob * self.num_prob_bins - 1e-6).long()
                    ]
                    + 1e-10
                )
            )
        else:
            loss_weighted = loss / (
                self.prob_bins_ema[torch.floor(pred_prob * self.num_prob_bins).long()]
                + 1e-10
            )
        loss = torch.mean(loss_weighted)

        prob_bins = torch.histc(pred_prob, bins=self.num_prob_bins, min=0, max=1).to(
            self.device
        )
        prob_bins = prob_bins / (torch.sum(prob_bins) + 1e-10) * self.num_prob_bins
        self.prob_bins_ema = (
            self.prob_bins_ema * self.alpha + (1 - self.alpha) * prob_bins
        )
        self.prob_bins_ema = (
            self.prob_bins_ema
            / (torch.sum(self.prob_bins_ema) + 1e-10)
            * self.num_prob_bins
        )

        if not self.enable_prob_input:
            classes = torch.histc(
                target.float(), bins=self.num_classes, min=0, max=self.num_classes - 1
            ).to(self.device)
            classes = classes / (torch.sum(classes) + 1e-10) * self.num_classes
            self.classes_ema = (
                self.classes_ema * self.alpha + (1 - self.alpha) * classes
            )
            self.classes_ema = (
                self.classes_ema
                / (torch.sum(self.classes_ema) + 1e-10)
                * self.num_classes
            )

        return loss
