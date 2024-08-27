from torch import nn

from .conv_next import ConvNextBlock

__all__ = ["ConvNextBlock"]


def get_blocks(name, num_blocks, num_dims, block_kwargs):
    if num_blocks == 0:
        return nn.Identity()
    assert num_blocks == len(block_kwargs)

    blocks = nn.Sequential(
        *[
            globals()[name](num_dims=num_dims, **block_kwargs[i])
            for i in range(num_blocks)
        ]
    )
    return blocks
