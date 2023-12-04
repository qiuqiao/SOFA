import torch.nn as nn


class ResidualBasicBlock(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=None, n_groups=16):
        super(ResidualBasicBlock, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = (
            hidden_dims
            if hidden_dims is not None
            else max(n_groups * (output_dims // n_groups), n_groups)
        )
        self.n_groups = n_groups

        self.block = nn.Sequential(
            nn.Conv1d(
                self.input_dims,
                self.hidden_dims,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            nn.Hardswish(),
            nn.Conv1d(
                self.hidden_dims,
                self.output_dims,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        self.shortcut = nn.Sequential(
            nn.Linear(self.input_dims, self.output_dims, bias=False)
            if self.input_dims != self.output_dims
            else nn.Identity()
        )

        self.out = nn.Sequential(
            nn.LayerNorm(self.output_dims),
            nn.Hardswish(),
        )

    def forward(self, x):
        x = self.block(x.transpose(1, 2)).transpose(1, 2) + self.shortcut(x)
        x = self.out(x)
        return x


class ResidualBottleNeckBlock(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=None, n_groups=16):
        super(ResidualBottleNeckBlock, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = (
            hidden_dims
            if hidden_dims is not None
            else max(n_groups * ((output_dims // 4) // n_groups), n_groups)
        )
        self.n_groups = n_groups

        self.input_proj = nn.Linear(self.input_dims, self.hidden_dims, bias=False)
        self.conv = nn.Sequential(
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            nn.Hardswish(),
            nn.Conv1d(
                self.hidden_dims,
                self.hidden_dims,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(self.n_groups, self.hidden_dims),
            nn.Hardswish(),
        )
        self.output_proj = nn.Linear(self.hidden_dims, self.output_dims, bias=False)

        self.shortcut = nn.Sequential(
            nn.Linear(self.input_dims, self.output_dims)
            if self.input_dims != self.output_dims
            else nn.Identity()
        )

        self.out = nn.Sequential(
            nn.LayerNorm(self.output_dims),
            nn.Hardswish(),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.output_proj(h)
        return self.out(h + self.shortcut(x))
