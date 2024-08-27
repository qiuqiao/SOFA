import torch
from torch import nn

from ..activation import get_activation
from ..normalization import get_normalization


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        num_dims=64,
        expansion=4,
        kernel_size=7,
        dilation=1,
        bias=True,
        layer_scale=True,
        norm_layer="RMSNorm",
        activation="SwiGLU",
    ):
        super(ConvNextBlock, self).__init__()

        self.dw_conv = nn.Conv1d(
            in_channels=num_dims,
            out_channels=num_dims,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            padding="same",
        )
        self.norm = get_normalization(norm_layer, num_dims)
        self.pw_conv1 = nn.Conv1d(
            in_channels=num_dims,
            out_channels=num_dims * expansion,
            kernel_size=1,
            bias=bias,
            padding="same",
        )
        self.activation = get_activation(activation, num_dims * expansion)
        self.pw_conv2 = nn.Conv1d(
            in_channels=num_dims * expansion,
            out_channels=num_dims,
            kernel_size=1,
            bias=bias,
            padding="same",
        )
        self.gamma = (
            nn.Parameter(torch.full((1, num_dims, 1), 1e-5)) if layer_scale else None
        )

    def forward(self, x):
        residual = x

        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.activation(x)
        x = self.pw_conv2(x)
        x = x * self.gamma if self.gamma is not None else x

        return x + residual
