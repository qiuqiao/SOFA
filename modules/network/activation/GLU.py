import torch
from einops import rearrange
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.projection = (
            nn.Conv1d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        # input: Tensor[batch_size seq_len, hidden_dims]
        # output: Tensor[batch_size seq_len, hidden_dims]
        gate = torch.sigmoid(self.linear(x))
        output = self.projection(x) * gate
        return output
