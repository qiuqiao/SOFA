import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, num_dims, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_dims, 1))

    def forward(self, x):
        norm = (torch.mean(x**2, dim=-2, keepdim=True) ** (0.5)) * self.weight
        return x / (norm + self.eps)


if __name__ == "__main__":
    rms_norm = RMSNorm(64)
    x = torch.randn(4, 64, 1024)
    y = rms_norm(x)
    print(y.shape, y.mean(), y.std())

    y.sum().backward()
