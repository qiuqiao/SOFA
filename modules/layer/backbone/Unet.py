import torch
import torch.nn as nn
from einops import rearrange, repeat
from modules.layer.block.conformer import ForwardBackwardConformerBlock
from modules.layer.block.residual import Residual


class UNetBackbone(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, block=ForwardBackwardConformerBlock, **kwargs):
        super(UNetBackbone, self).__init__()

        self.encoder_1 = block(input_dims, hidden_dims, hidden_dims, **kwargs)
        self.residual_e1 = Residual(input_dims, hidden_dims)
        self.encoder_2 = block(2 * hidden_dims, 2 * hidden_dims, 2 * hidden_dims, **kwargs)
        self.encoder_3 = block(4 * hidden_dims, 4 * hidden_dims, 4 * hidden_dims, **kwargs)
        self.encoder_4 = block(8 * hidden_dims, 8 * hidden_dims, 8 * hidden_dims, **kwargs)
        self.encoder_5 = block(16 * hidden_dims, 16 * hidden_dims, 16 * hidden_dims, **kwargs)

        self.bottle_neck = block(32 * hidden_dims, 32 * hidden_dims, 32 * hidden_dims, **kwargs)

        self.decoder_5 = block((32 + 16) * hidden_dims, 16 * hidden_dims, **kwargs)
        self.residual_d5 = Residual((32 + 16) * hidden_dims, 16 * hidden_dims)
        self.decoder_4 = block((16 + 8) * hidden_dims, 8 * hidden_dims, **kwargs)
        self.residual_d4 = Residual((16 + 8) * hidden_dims, 8 * hidden_dims)
        self.decoder_3 = block((8 + 4) * hidden_dims, 4 * hidden_dims, **kwargs)
        self.residual_d3 = Residual((8 + 4) * hidden_dims, 4 * hidden_dims)
        self.decoder_2 = block((4 + 2) * hidden_dims, 2 * hidden_dims, **kwargs)
        self.residual_d2 = Residual((4 + 2) * hidden_dims, 2 * hidden_dims)
        self.decoder_1 = block((2 + 1) * hidden_dims, hidden_dims, **kwargs)
        self.residual_d1 = Residual((2 + 1) * hidden_dims, hidden_dims)

        self.out = block(input_dims + hidden_dims, output_dims, hidden_dims, **kwargs)

    def forward(self, x):
        x0 = x

        x1 = x = self.residual_e1(x, self.encoder_1(x))

        x = rearrange(x, "B (T1 T2) C  -> B T1 (C T2)", T2=2)  # Patch Merging #TODO C和T的顺序要调换
        x2 = x = self.encoder_2(x) + x

        x = rearrange(x, "B (T1 T2) C  -> B T1 (C T2)", T2=2)
        x3 = x = self.encoder_3(x) + x

        x = rearrange(x, "B (T1 T2) C  -> B T1 (C T2)", T2=2)
        x4 = x = self.encoder_4(x) + x

        x = rearrange(x, "B (T1 T2) C  -> B T1 (C T2)", T2=2)
        x5 = x = self.encoder_5(x) + x

        x = rearrange(x, "B (T1 T2) C  -> B T1 (C T2)", T2=2)
        x = self.bottle_neck(x)

        x = repeat(x, "B T C -> B (T T2) C", T2=2)
        x = torch.cat((x, x5), dim=-1)
        x = self.residual_d5(x, self.decoder_5(x))

        x = repeat(x, "B T C -> B (T T2) C", T2=2)
        x = torch.cat((x, x4), dim=-1)
        x = self.residual_d4(x, self.decoder_4(x))

        x = repeat(x, "B T C -> B (T T2) C", T2=2)
        x = torch.cat((x, x3), dim=-1)
        x = self.residual_d3(x, self.decoder_3(x))

        x = repeat(x, "B T C -> B (T T2) C", T2=2)
        x = torch.cat((x, x2), dim=-1)
        x = self.residual_d2(x, self.decoder_2(x))

        x = repeat(x, "B T C -> B (T T2) C", T2=2)
        x = torch.cat((x, x1), dim=-1)
        x = self.residual_d1(x, self.decoder_1(x))

        print(x.shape, x0.shape)
        x = torch.cat((x, x0), dim=-1)
        out = self.out(x)

        return out


if __name__ == "__main__":
    pass
    # model = UNetBackbone(1, 2, 64)
    # print(model)
    # x = torch.randn(16, 320, 1)
    # out = model(x)
    # print(x.shape, out.shape)
