class UNetBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, block=ForwardBackwardConformerBlock):
        super(UNetBackbone, self).__init__()

        self.encoder_1 = block(in_channels=in_channels, out_channels=hidden_channels)
        self.residual_e1 = Residual(in_channels, hidden_channels)
        self.encoder_2 = block(in_channels=2 * hidden_channels, out_channels=2 * hidden_channels)
        self.encoder_3 = block(in_channels=4 * hidden_channels, out_channels=4 * hidden_channels)
        self.encoder_4 = block(in_channels=8 * hidden_channels, out_channels=8 * hidden_channels)
        self.encoder_5 = block(in_channels=16 * hidden_channels, out_channels=16 * hidden_channels)

        self.bottle_neck = block(in_channels=32 * hidden_channels, out_channels=32 * hidden_channels)

        self.decoder_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder_5 = block(in_channels=(32 + 16) * hidden_channels, out_channels=16 * hidden_channels)
        self.residual_d5 = Residual((32 + 16) * hidden_channels, 16 * hidden_channels)
        self.decoder_4 = block(in_channels=(16 + 8) * hidden_channels, out_channels=8 * hidden_channels)
        self.residual_d4 = Residual((16 + 8) * hidden_channels, 8 * hidden_channels)
        self.decoder_3 = block(in_channels=(8 + 4) * hidden_channels, out_channels=4 * hidden_channels)
        self.residual_d3 = Residual((8 + 4) * hidden_channels, 4 * hidden_channels)
        self.decoder_2 = block(in_channels=(4 + 2) * hidden_channels, out_channels=2 * hidden_channels)
        self.residual_d2 = Residual((4 + 2) * hidden_channels, 2 * hidden_channels)
        self.decoder_1 = block(in_channels=(2 + 1) * hidden_channels, out_channels=hidden_channels)
        self.residual_d1 = Residual((2 + 1) * hidden_channels, hidden_channels)

        self.out = block(in_channels=in_channels + hidden_channels, out_channels=out_channels)

    def forward(self, x):
        x0 = x

        x1 = x = self.residual_e1(x, self.encoder_1(x))

        x = rearrange(x, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x2 = x = self.encoder_2(x) + x

        x = rearrange(x, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x3 = x = self.encoder_3(x) + x

        x = rearrange(x, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x4 = x = self.encoder_4(x) + x

        x = rearrange(x, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x5 = x = self.encoder_5(x) + x

        x = rearrange(x, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x = self.bottle_neck(x)

        x = self.decoder_upsample(x)
        x = torch.cat((x, x5), dim=1)
        x = self.residual_d5(x, self.decoder_5(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x4), dim=1)
        x = self.residual_d4(x, self.decoder_4(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x3), dim=1)
        x = self.residual_d3(x, self.decoder_3(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x2), dim=1)
        x = self.residual_d2(x, self.decoder_2(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x1), dim=1)
        x = self.residual_d1(x, self.decoder_1(x))

        x = torch.cat((x, x0), dim=1)
        out = self.out(x)

        return out
