import torch.nn as nn


class BaseDowmSampling(nn.Module):
    def __init__(self, input_dims, output_dims, down_sampling_factor=2):
        super(BaseDowmSampling, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class BaseUpSampling(nn.Module):
    def __init__(self, input_dims, output_dims, up_sampling_factor=2):
        super(BaseUpSampling, self).__init__()

    def forward(self, x):
        raise NotImplementedError
