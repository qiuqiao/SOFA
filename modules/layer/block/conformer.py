import torch
import torch.nn as nn
from einops import rearrange
from .attention import MultiHeadSelfAttention
from .func_module import FuncModule
from .residual import Residual
from modules.layer.activation.GLU import GLU


class ConformerBlock(nn.Module):
    def __init__(
            self,
            input_dims=128,
            output_dims=128,
            hidden_dims=64,
            kernel_size=3,
            dropout=0.1,
            num_heads=8,
            max_seq_len=3200,
            mask="none",
    ):
        super(ConformerBlock, self).__init__()
        self.feed_forward_1 = nn.Sequential(
            nn.LayerNorm(input_dims),  # nn.LayerNorm(input_dims),
            nn.Linear(input_dims, hidden_dims),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Dropout(dropout),
        )
        self.multi_head_self_attention = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            MultiHeadSelfAttention(hidden_dims, num_heads=num_heads, mask=mask, max_seq_len=max_seq_len),
        )
        self.convolution = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            FuncModule(lambda x: rearrange(x, "b t c -> b c t")),
            nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dims,
            ),
            FuncModule(lambda x: rearrange(x, "b c t -> b t c")),
            GLU(hidden_dims, hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            # nn.Conv1d(hidden_dims, hidden_dims, 1, 1, 0, 1, 1),
            nn.LayerNorm(hidden_dims),
            nn.Hardswish(),
            FuncModule(lambda x: rearrange(x, "b t c -> b c t")),
            nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dims,
            ),
            FuncModule(lambda x: rearrange(x, "b c t -> b t c")),
            nn.Dropout(dropout),
        )
        self.feed_forward_2 = nn.Sequential(
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, output_dims),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(output_dims)

        self.residual_i_h = Residual(input_dims, hidden_dims)
        self.residual_h_h = Residual(hidden_dims, hidden_dims)
        self.residual_h_o = Residual(hidden_dims, output_dims)

    def forward(self, x):
        x = self.residual_i_h(x, (1 / 2) * self.feed_forward_1(x))
        # Multi-head self-attention
        x = self.residual_h_h(x, self.multi_head_self_attention(x))
        # Convolution
        x = self.residual_h_h(x, self.convolution(x))
        # Feed Forward
        x = self.residual_h_o(x, (1 / 2) * self.feed_forward_2(x))
        # Norm
        x = self.norm(x)
        return x


class ForwardBackwardConformerBlock(nn.Module):
    def __init__(
            self,
            input_dims=128,
            output_dims=128,
            hidden_dims=128,
            kernel_size=3,
            dropout=0.1,
            num_heads=8,
            max_seq_len=3200,
    ):
        super(ForwardBackwardConformerBlock, self).__init__()
        self.forward_block = ConformerBlock(input_dims,
                                            hidden_dims,
                                            hidden_dims,
                                            kernel_size,
                                            dropout,
                                            num_heads,
                                            max_seq_len,
                                            mask='upper',
                                            )
        self.backward_block = ConformerBlock(hidden_dims,
                                             output_dims,
                                             hidden_dims,
                                             kernel_size,
                                             dropout,
                                             num_heads,
                                             max_seq_len,
                                             mask='lower',
                                             )

    def forward(self, x):
        x = self.forward_block(x)
        x = self.backward_block(x)
        return x


if __name__ == "__main__":
    # test
    # bs, l, dims = 16, 320, 128
    # input_tensor = torch.randn(bs, l, dims)
    # model = ForwardBackwardConformerBlock(dims, 64)
    # y = model(input_tensor)
    # print(input_tensor.shape, y.shape)
    pass
