import torch
import torch.nn as nn

from modules.network.block.resnet_block import ResidualBasicBlock
from modules.network.scaling.base import BaseDowmSampling, BaseUpSampling
from modules.network.scaling.stride_conv import DownSampling, UpSampling


class UNet(nn.Module):
    def __init__(
        self,
        n_dims,
        down_sampling_factor=2,
        down_sampling_times=5,
        channels_scaleup_factor=2,
        **kwargs
    ):
        """_summary_

        Args:
            n_dims (int): input and output feature dimension
            down_sampling_factor (int, optional): down sampling factor. Defaults to 2.
            down_sampling_times (int, optional): down sampling times. Defaults to 5.
            channels_scaleup_factor (int, optional): channels scale up factor. Defaults to 2.
            **kwargs: other parameters for ResidualBasicBlock
        """
        super(UNet, self).__init__()

        self.n_dims = n_dims
        self.divisible_factor = down_sampling_factor**down_sampling_times

        self.encoders = nn.ModuleList()
        for i in range(down_sampling_times - 1):
            i += 1
            self.encoders.append(
                nn.Sequential(
                    DownSampling(
                        int(channels_scaleup_factor ** (i - 1)) * n_dims,
                        int(channels_scaleup_factor**i) * n_dims,
                        down_sampling_factor,
                    ),
                    ResidualBasicBlock(
                        int(channels_scaleup_factor**i) * n_dims,
                        int(channels_scaleup_factor**i) * n_dims,
                        **kwargs
                    ),
                )
            )

        self.bottle_neck = nn.Sequential(
            DownSampling(
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * n_dims,
                int(channels_scaleup_factor**down_sampling_times) * n_dims,
                down_sampling_factor,
            ),
            ResidualBasicBlock(
                int(channels_scaleup_factor**down_sampling_times) * n_dims,
                int(channels_scaleup_factor**down_sampling_times) * n_dims,
                **kwargs
            ),
            UpSampling(
                int(channels_scaleup_factor**down_sampling_times) * n_dims,
                int(channels_scaleup_factor ** (down_sampling_times - 1)) * n_dims,
                down_sampling_factor,
            ),
        )

        self.decoders = nn.ModuleList()
        for i in range(down_sampling_times - 1):
            i += 1
            self.decoders.append(
                nn.Sequential(
                    ResidualBasicBlock(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * n_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * n_dims,
                        **kwargs
                    ),
                    UpSampling(
                        int(channels_scaleup_factor ** (down_sampling_times - i))
                        * n_dims,
                        int(channels_scaleup_factor ** (down_sampling_times - i - 1))
                        * n_dims,
                        down_sampling_factor,
                    ),
                )
            )

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): shape: (B, T, C)

        Returns:
            torch.Tensor: shape: (B, T, C)
        """
        T = x.shape[1]
        padding_len = T % self.divisible_factor
        if padding_len != 0:
            x = nn.functional.pad(x, (0, 0, 0, self.divisible_factor - padding_len))

        h = [x]
        for encoder in self.encoders:
            h.append(encoder(h[-1]))

        h_ = [self.bottle_neck(h[-1])]
        for i, decoder in enumerate(self.decoders):
            h_.append(decoder(h_[-1] + h[-1 - i]))

        out = h_[-1]
        out = out[:, :T, :]

        return out


if __name__ == "__main__":
    # pass
    model = UNet(64)
    print(model)
    x = torch.randn(16, 320, 64)
    out = model(x)
    print(x.shape, out.shape)
