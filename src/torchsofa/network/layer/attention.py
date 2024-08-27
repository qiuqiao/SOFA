from math import log

import torch
from einops import rearrange
from torch import nn


class EntropyInvarianceAttention(nn.Module):
    """
    Scaled Dot-Product Attention with Entropy Invariance.
    https://kexue.fm/archives/8823
    """

    def __init__(self, dim, avg_k_length=20, num_heads=1):
        super(EntropyInvarianceAttention, self).__init__()

        self.scale = 1 / ((dim**0.5) * log(avg_k_length))

        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v=None, k_length=None, return_attn_matrix=False):
        """
        Args:
            q : (batch_size, num_heads * head_dim, q_length)
            k : (batch_size, num_heads * head_dim, k_length)
            v : (batch_size, num_heads * v_dim, k_length)
            seq_length : (batch_size). Defaults to None.

        Returns:
            output : (batch_size, num_heads * v_dim, q_length) if return_attn_matrix is False
            attn_matrix : (batch_size, num_heads, q_length, k_length) if return_attn_matrix is True
        """
        q = rearrange(q, "b (h d) l -> b h d l", h=self.num_heads)
        k = rearrange(k, "b (h d) l -> b h d l", h=self.num_heads)
        v = rearrange(v, "b (h d) l -> b h d l", h=self.num_heads)

        if k_length is None:
            k_length = torch.full((k.size(0), 1, 1, 1), k.size(-1), device=k.device)
        else:
            k_length = rearrange(k_length, "b -> b 1 1 1")

        attn = torch.einsum("bhdq,bhdk->bhqk", q, k)
        attn = (self.scale * torch.log(k_length)) * attn
        attn = self.softmax(attn)

        if return_attn_matrix:
            return attn

        output = torch.einsum("bhqk,bhdk->bhdq", attn, v)
        output = rearrange(output, "b h d l -> b (h d) l")
        return output


if __name__ == "__main__":
    attn = EntropyInvarianceAttention(16, avg_k_length=12, num_heads=2)
    q = torch.randn(4, 16, 10)
    k = torch.randn(4, 16, 12)
    v = torch.randn(4, 32, 12)

    matrix = attn(q, k, v, return_attn_matrix=True)
    out = attn(q, k, v)

    print(matrix.shape)
    print(out.shape)

    import matplotlib.pyplot as plt

    plt.imshow(matrix[0, 0].detach().cpu().numpy())
    plt.show()
