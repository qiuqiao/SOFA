import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, num_dims, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_dims, 1))

    def forward(self, x):
        norm = x.norm(2, dim=-2, keepdim=True) * self.weight
        return x / (norm + self.eps)
