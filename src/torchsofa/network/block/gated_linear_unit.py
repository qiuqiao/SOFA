import torch
from torch import nn

from ..activation import get_activation
from ..layer.attention import EntropyInvarianceAttention
from ..normalization import get_normalization


class SoftmaxGAU(nn.Module):
    """
    https://kexue.fm/archives/9019
    """

    def __init__(
        self,
        num_dims=64,
        expansion=4,
        num_heads=1,
        head_dims=32,
        avg_length=20,
        norm_layer="RMSNorm",
        activation="SiLU",
    ):
        super().__init__()
        self.mlp_in = nn.Conv1d(
            in_channels=num_dims,
            out_channels=2 * expansion * num_dims,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.activation_in = get_activation(activation, 2 * expansion * num_dims)

        self.num_heads = num_heads
        self.head_dims = head_dims
        self.mlp_attn = nn.Conv1d(
            in_channels=num_dims,
            out_channels=num_heads * head_dims,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.activation_attn = get_activation(activation, num_heads * head_dims)
        self.affine_attn = nn.Conv1d(
            in_channels=2 * num_heads * head_dims,
            out_channels=2 * num_heads * head_dims,
            kernel_size=1,
            bias=True,
            padding="same",
            groups=2 * num_heads * head_dims,
        )
        self.attention = EntropyInvarianceAttention(
            dim=head_dims, avg_k_length=avg_length, num_heads=num_heads
        )

        self.mlp_out = nn.Conv1d(
            in_channels=expansion * num_dims,
            out_channels=num_dims,
            kernel_size=1,
            bias=True,
            padding="same",
        )

        self.norm = get_normalization(norm_layer, num_dims)

    def forward(self, x):
        residual = x

        u, v = self.activation_in(self.mlp_in(x)).chunk(2, dim=1)

        z = self.mlp_attn(x)
        z = torch.cat((z, z), dim=1)
        q, k = self.affine_attn(z).chunk(2, dim=1)
        v = self.attention(q, k, v)

        out = self.mlp_out(u * v)

        return self.norm(residual + out)


if __name__ == "__main__":
    model = SoftmaxGAU(128)
    x = torch.randn(4, 128, 20)

    y = model(x)
    print(y.shape, y.mean(), y.std())
