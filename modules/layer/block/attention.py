import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            mask: str = "none",
            init_type: str = "xavier_uniform",
    ):
        super().__init__()

        assert num_heads > 0, "num_heads must be positive"
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        assert (
                       model_dim // num_heads
               ) % 2 == 0, "model_dim // num_heads must be divisible by 2"
        assert max_seq_len > 0, "max_seq_len must be positive"
        assert 0.0 <= dropout < 1.0, "dropout must be in range [0.0, 1.0)"
        assert mask in [
            "none",
            "upper",
            "lower",
        ], "mask must be one of [none, upper, lower]"
        assert init_type in [
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        ], "init_type must be one of [xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal]"

        self.model_dim = model_dim
        self.d_k = model_dim // num_heads
        self.num_heads = num_heads
        self.mask = mask

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.rotation_complex_matrix = self.precompute_rotation_complex_matrix(
            self.d_k, max_seq_len
        )

        self.init_type = init_type
        self.apply(self.init_weights)

    def init_weights(self, module):
        init_type = self.init_type
        if isinstance(module, nn.Linear):
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def precompute_rotation_complex_matrix(
            dim: int, seq_len: int, theta_base=torch.Tensor([10000.0])
    ):
        power = torch.arange(dim // 2) * (-2) / dim
        theta_vector = torch.pow(theta_base, power).float()
        position_vector = torch.arange(seq_len).float()
        rotation_angle_matrix = torch.outer(position_vector, theta_vector)
        rotation_complex_matrix = (
            torch.polar(torch.ones_like(rotation_angle_matrix), rotation_angle_matrix)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return rotation_complex_matrix

    @staticmethod
    def apply_rotary_emb(
            xq: torch.Tensor, xk: torch.Tensor, rotation_complex_matrix: torch.Tensor
    ):
        # xq.shape = [batch_size, num_heads, seq_len, d_k]
        # xq_complex.shape = [batch_size, num_heads, seq_len, d_k // 2, 2]
        xq_complex = rearrange(xq, "b h t (d c) -> b h t d c", c=2)
        xk_complex = rearrange(xk, "b h t (d c) -> b h t d c", c=2)

        xq_complex = torch.view_as_complex(xq_complex)
        xk_complex = torch.view_as_complex(xk_complex)

        rotation_complex_matrix = rotation_complex_matrix[:, :, xq_complex.shape[-2], :]
        xq_out = torch.view_as_real(xq_complex * rotation_complex_matrix)
        xk_out = torch.view_as_real(xk_complex * rotation_complex_matrix)

        xq_out = rearrange(xq_out, "b h t d c -> b h t (d c)", c=2)
        xk_out = rearrange(xk_out, "b h t d c -> b h t (d c)", c=2)

        return xq_out, xk_out

    def forward(self, x: torch.Tensor):
        # input: Tensor[batch_size seq_len, hidden_dims]
        # output: Tensor[batch_size seq_len, hidden_dims]
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, self.num_heads, seq_len, self.d_k)
        xk = xk.view(batch_size, self.num_heads, seq_len, self.d_k)
        xv = xv.view(batch_size, self.num_heads, seq_len, self.d_k)

        xq, xk = self.apply_rotary_emb(xq, xk, self.rotation_complex_matrix)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / np.sqrt(
            self.d_k
        )  # size: (batch_size, num_heads, seq_len, seq_len)
        if self.mask == "upper":
            mask = torch.triu(torch.ones_like(scores[0, 0]), diagonal=1)
            scores.masked_fill_(mask == 1, -1e9)
        elif self.mask == "lower":
            mask = torch.tril(torch.ones_like(scores[0, 0]), diagonal=-1)
            scores.masked_fill_(mask == 1, -1e9)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, xv)  # size: (batch_size, num_heads, seq_len, d_k)
        output = rearrange(output, "b h t d -> b t (h d)")

        output = self.linear(output)  # size: (batch_size, seq_len, model_dim)
        return output
