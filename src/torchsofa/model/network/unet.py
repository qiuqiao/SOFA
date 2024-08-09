from math import prod

from torch import nn

from .block import get_blocks


class Unet(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        downsample_rates=(1, 3, 3, 3, 3),
        block_type="ConvNextBlock",
        num_blocks=(1, 2, 2, 2, 1),
        hidden_dims=(128, 128, 128, 128, 256),
        block_kwargs={},
        bottleneck_type="ConvNextBlock",
        bottleneck_hidden_dims=512,
        bottleneck_kwargs={},
    ):
        super(Unet, self).__init__()

        # check downsample_rates
        assert isinstance(downsample_rates, tuple), "downsample_rates must be a tuple"

        # check num_blocks
        if isinstance(num_blocks, int):
            num_blocks = (num_blocks,) * len(downsample_rates)
        elif isinstance(num_blocks, tuple):
            assert len(downsample_rates) == len(
                num_blocks
            ), f"downsample_rates and num_blocks must have the same length: {len(downsample_rates)} != {len(num_blocks)}"
        else:
            raise TypeError(f"num_blocks must be an int or a tuple: {type(num_blocks)}")

        # check hidden_dims
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,) * len(downsample_rates)
        if isinstance(hidden_dims, tuple):
            assert len(downsample_rates) == len(
                hidden_dims
            ), f"downsample_rates and hidden_dims must have the same length: {len(downsample_rates)}!= {len(hidden_dims)}"
        else:
            raise TypeError(
                f"hidden_dims must be an int or a tuple: {type(hidden_dims)}"
            )

        # check block_kwargs
        if isinstance(block_kwargs, dict):
            block_kwargs = (block_kwargs,) * len(downsample_rates)
        if isinstance(block_kwargs, tuple):
            assert len(downsample_rates) == len(
                block_kwargs
            ), f"downsample_rates and block_kwargs must have the same length: {len(downsample_rates)}!= {len(block_kwargs)}"
            block_kwargs = list(block_kwargs)
            for i in range(len(block_kwargs)):
                if isinstance(block_kwargs[i], dict):
                    block_kwargs[i] = (block_kwargs[i],) * num_blocks[i]
                elif isinstance(block_kwargs[i], tuple):
                    assert (
                        len(block_kwargs[i]) == num_blocks[i]
                    ), f"block_kwargs[{i}] must have the same length as num_blocks[{i}]: {len(block_kwargs[i])}!={num_blocks[i]}"
                else:
                    raise TypeError(
                        f"block_kwargs[{i}] must be a dict or a tuple: {type(block_kwargs[i])}"
                    )
        else:
            raise TypeError(
                f"block_kwargs must be a dict or a tuple: {type(block_kwargs)}"
            )

        self.divisable_factor = prod(downsample_rates)

        # print(downsample_rates)
        # print(num_blocks)
        # print(hidden_dims)
        # print(block_kwargs)

        # model
        last_dim = input_dims
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx, (downsample_rate, num_block, hidden_dim, block_kwarg) in enumerate(
            zip(downsample_rates, num_blocks, hidden_dims, block_kwargs)
        ):
            self.encoder.append(
                nn.Sequential(
                    (
                        nn.Conv1d(last_dim, hidden_dim, 1)
                        if last_dim != hidden_dim
                        else nn.Identity()
                    ),
                    (
                        nn.AvgPool1d(downsample_rate)
                        if downsample_rate > 1
                        else nn.Identity()
                    ),
                    get_blocks(block_type, num_block, hidden_dim, block_kwarg),
                )
            )

            if idx == 0:
                last_dim = output_dims
            self.decoder.append(
                nn.Sequential(
                    get_blocks(block_type, num_block, hidden_dim, block_kwarg),
                    (
                        nn.Upsample(scale_factor=downsample_rate)
                        if downsample_rate > 1
                        else nn.Identity()
                    ),
                    (
                        nn.Conv1d(hidden_dim, last_dim, 1)
                        if last_dim != hidden_dim
                        else nn.Identity()
                    ),
                )
            )

            last_dim = hidden_dim

        self.bottle_neck = nn.Sequential(
            (
                nn.Conv1d(last_dim, bottleneck_hidden_dims, 1)
                if last_dim != bottleneck_hidden_dims
                else nn.Identity()
            ),
            get_blocks(
                bottleneck_type, 1, bottleneck_hidden_dims, (bottleneck_kwargs,)
            ),
            (
                nn.Conv1d(bottleneck_hidden_dims, last_dim, 1)
                if output_dims != bottleneck_hidden_dims
                else nn.Identity()
            ),
        )

    def forward(self, x):
        length = x.shape[-1]
        if length % self.divisable_factor != 0:
            x = nn.functional.pad(
                x, (0, self.divisable_factor - x.shape[-1] % self.divisable_factor)
            )

        skip = []
        for encoder in self.encoder:
            x = encoder(x)
            skip.append(x)
            # print(x.shape)

        x = self.bottle_neck(x)

        for decoder in self.decoder[::-1]:
            x = x + skip.pop()
            x = decoder(x)

        return x[:, :, :length]


if __name__ == "__main__":
    # python -m src.torchsofa.model.network.unet
    import torch

    Unet(32, 64)
    print(Unet(32, 64))

    x = torch.randn(4, 32, 1000)
    y = Unet(32, 64)(x)
    print(y.shape)

    y.sum().backward()
