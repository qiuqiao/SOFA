import torch.nn as nn


class FuncModule(nn.Module):
    def __init__(self, func):
        super(FuncModule, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
