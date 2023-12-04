import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        max_seq_len: int = 3200,
        dropout: float = 0.0,
        mask: str = "none",
        init_type: str = "kaiming_uniform",
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

        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.d_k = model_dim // num_heads
        self.num_heads = num_heads
        self.mask = mask

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "theta_base", torch.Tensor([10000.0])
        )  # 常量tensor需要用register_buffer注册，否则.to(device)不起作用
        cos, sin = self.precompute_rotation_matrix(self.max_seq_len, self.theta_base)
        self.register_buffer("rotation_matrix_cos", cos)
        self.register_buffer("rotation_matrix_sin", sin)

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

    def precompute_rotation_matrix(self, seq_len: int, theta_base):
        dim = self.d_k
        power = torch.arange(dim // 2) * (-2) / dim
        theta_vector = torch.pow(theta_base, power)
        position_vector = torch.arange(seq_len)
        rotation_angle_matrix = torch.outer(position_vector, theta_vector)
        rotation_angle_matrix = repeat(
            rotation_angle_matrix, "l d -> l (d repeat)", repeat=2
        )
        rotation_matrix_cos = torch.cos(rotation_angle_matrix).unsqueeze(0).unsqueeze(0)
        rotation_matrix_sin = torch.sin(rotation_angle_matrix).unsqueeze(0).unsqueeze(0)
        return rotation_matrix_cos, rotation_matrix_sin

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor):
        # xq.shape = [batch_size, num_heads, seq_len, d_k]
        def get_sin_weight(q):
            q = rearrange(q, "b h t (d1 d2) -> b h t d2 d1", d2=2)
            q = q.clone()
            q[:, :, :, 1, :] = -1 * q[:, :, :, 1, :]
            q = q[:, :, :, [1, 0], :]
            q = rearrange(q, "b h t d2 d1 -> b h t (d1 d2)")
            return q

        # print(xq.shape, self.rotation_matrix_cos.shape)
        xq_ = get_sin_weight(xq)
        xk_ = get_sin_weight(xk)
        xq_out = (
            xq
            * self.rotation_matrix_cos[
                :,
                :,
                xq.shape[2],
                :,
            ]
            + xq_
            * self.rotation_matrix_sin[
                :,
                :,
                xq.shape[2],
                :,
            ]
        )
        xk_out = (
            xk
            * self.rotation_matrix_cos[
                :,
                :,
                xk.shape[2],
                :,
            ]
            + xk_
            * self.rotation_matrix_sin[
                :,
                :,
                xk.shape[2],
                :,
            ]
        )

        return xq_out, xk_out

    def _update_RoPE(self, seq_len):
        cos, sin = self.precompute_rotation_matrix(seq_len, self.theta_base)
        self.cos = cos
        self.sin = sin
        self.max_seq_len = seq_len

    def forward(self, x: torch.Tensor):  # , lengths=None
        # input: Tensor[batch_size seq_len, hidden_dims], lengths: Tensor[batch_size]
        # output: Tensor[batch_size seq_len, hidden_dims]
        batch_size, seq_len, _ = x.shape

        if self.max_seq_len < seq_len:
            self._update_RoPE(seq_len)

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, self.num_heads, seq_len, self.d_k)
        xk = xk.view(batch_size, self.num_heads, seq_len, self.d_k)
        xv = xv.view(batch_size, self.num_heads, seq_len, self.d_k)

        xq, xk = self.apply_rotary_emb(xq, xk)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / np.sqrt(
            self.d_k
        )  # size: (batch_size, num_heads, seq_len, seq_len)
        if self.mask == "upper":
            mask = torch.triu(torch.ones_like(scores[0, 0]).float(), diagonal=1)
            scores.masked_fill_(mask == 1, -1e9)
        elif self.mask == "lower":
            mask = torch.tril(torch.ones_like(scores[0, 0]).float(), diagonal=-1)
            scores.masked_fill_(mask == 1, -1e9)
        # if lengths is not None:
        #     mask = torch.arange(seq_len).to(x.device)[None, :] >= lengths[:, None]
        #     scores.masked_fill_(mask[:, None, None, :] == 1, -1e9)
        #     scores.masked_fill_(mask[:, None, :, None] == 1, -1e9)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, xv)  # size: (batch_size, num_heads, seq_len, d_k)
        output = rearrange(output, "b h t d -> b t (h d)")

        output = self.linear(output)  # size: (batch_size, seq_len, model_dim)
        return output


if __name__ == "__main__":
    model = MultiHeadSelfAttention(128, 8)
    tensor_x = torch.randn(4, 32, 128)
    y = model(tensor_x)
    print(y.shape)
