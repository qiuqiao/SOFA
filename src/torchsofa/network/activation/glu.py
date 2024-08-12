import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, num_dims):
        super().__init__()
        self.proj = nn.Linear(num_dims, num_dims * 2)
        self.activation = nn.SiLU()

    def forward(self, x):
        x, gate = self.activation(self.proj(x.transpose(-1, -2))).chunk(2, dim=-1)
        return (x * gate).transpose(-1, -2)


if __name__ == "__main__":
    glu = SwiGLU(128)
    x = torch.randn(4, 128, 1024)
    y = glu(x)
    print(y.shape)
    y.sum().backward()
