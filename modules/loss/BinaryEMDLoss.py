import torch


class BinaryEMDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, target):
        # pred, target: [B,T]
        loss = self.loss(pred.cumsum(dim=-1), target.cumsum(dim=-1))
        loss += self.loss(
            pred.flip([-1]).cumsum(dim=-1), target.flip([-1]).cumsum(dim=-1)
        )
        return loss / 2
