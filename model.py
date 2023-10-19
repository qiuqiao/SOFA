import torch
import torch.nn as nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, dim_x, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.dim_x = dim_x
        self.projection = (
            nn.Conv1d(dim_x, dim_out, 1) if dim_x != dim_out else nn.Identity()
        )

    def forward(self, x, out):
        if x.shape[1] != self.dim_x:
            raise ValueError(
                f"Dimension mismatch: expected input dimension {self.dim_x}, but got {x.shape[1]}."
            )
        if out.shape[1] != self.dim_out:
            raise ValueError(
                f"Dimension mismatch: expected output dimension {self.dim_out}, but got {out.shape[1]}."
            )
        return out + self.projection(x)


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=128,
        hidden_dim_scaling=1,
        init_type="xavier_normal",
    ):
        super(UNetEncoder, self).__init__()

        # Encoder
        self.encoder_l1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                int(64 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(int(64 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(64 * hidden_dim_scaling + 0.5),
                int(64 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.BatchNorm1d(int(64 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_e1 = Residual(in_channels, int(64 * hidden_dim_scaling + 0.5))

        self.encoder_l2 = nn.Sequential(
            nn.Conv1d(
                int(128 * hidden_dim_scaling + 0.5),
                int(128 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(int(128 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(128 * hidden_dim_scaling + 0.5),
                int(128 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=5,
                dilation=5,
            ),
            nn.BatchNorm1d(int(128 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_e2 = Residual(
            int(128 * hidden_dim_scaling + 0.5), int(128 * hidden_dim_scaling + 0.5)
        )

        self.encoder_l3 = nn.Sequential(
            nn.Conv1d(
                int(256 * hidden_dim_scaling + 0.5),
                int(256 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(int(256 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(256 * hidden_dim_scaling + 0.5),
                int(256 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.BatchNorm1d(int(256 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_e3 = Residual(
            int(256 * hidden_dim_scaling + 0.5), int(256 * hidden_dim_scaling + 0.5)
        )

        self.encoder_l4 = nn.Sequential(
            nn.Conv1d(
                int(512 * hidden_dim_scaling + 0.5),
                int(512 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(int(512 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(512 * hidden_dim_scaling + 0.5),
                int(512 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=5,
                dilation=5,
            ),
            nn.BatchNorm1d(int(512 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_e4 = Residual(
            int(512 * hidden_dim_scaling + 0.5), int(512 * hidden_dim_scaling + 0.5)
        )

        self.encoder_l5 = nn.Sequential(
            nn.Conv1d(
                int(1024 * hidden_dim_scaling + 0.5),
                int(1024 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(int(1024 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(1024 * hidden_dim_scaling + 0.5),
                int(1024 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.BatchNorm1d(int(1024 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_e5 = Residual(
            int(1024 * hidden_dim_scaling + 0.5), int(1024 * hidden_dim_scaling + 0.5)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                int(2048 * hidden_dim_scaling + 0.5),
                int(2048 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(int(2048 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(2048 * hidden_dim_scaling + 0.5),
                int(2048 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=5,
                dilation=5,
            ),
            nn.BatchNorm1d(int(2048 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_bottleneck = Residual(
            int(2048 * hidden_dim_scaling + 0.5), int(2048 * hidden_dim_scaling + 0.5)
        )

        # Decoder
        self.decoder_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder_l1 = nn.Sequential(
            nn.Conv1d(
                int((64 + 128) * hidden_dim_scaling + 0.5),
                int(64 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(int(64 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(64 * hidden_dim_scaling + 0.5),
                out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.residual_d1 = Residual(
            int((64 + 128) * hidden_dim_scaling + 0.5), out_channels
        )

        self.decoder_l2 = nn.Sequential(
            nn.Conv1d(
                int((128 + 256) * hidden_dim_scaling + 0.5),
                int(128 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.BatchNorm1d(int(128 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(128 * hidden_dim_scaling + 0.5),
                int(128 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=5,
                dilation=5,
            ),
            nn.BatchNorm1d(int(128 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_d2 = Residual(
            int((128 + 256) * hidden_dim_scaling + 0.5),
            int(128 * hidden_dim_scaling + 0.5),
        )

        self.decoder_l3 = nn.Sequential(
            nn.Conv1d(
                int((256 + 512) * hidden_dim_scaling + 0.5),
                int(256 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(int(256 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(256 * hidden_dim_scaling + 0.5),
                int(256 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(int(256 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_d3 = Residual(
            int((256 + 512) * hidden_dim_scaling + 0.5),
            int(256 * hidden_dim_scaling + 0.5),
        )

        self.decoder_l4 = nn.Sequential(
            nn.Conv1d(
                int((512 + 1024) * hidden_dim_scaling + 0.5),
                int(512 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.BatchNorm1d(int(512 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(512 * hidden_dim_scaling + 0.5),
                int(512 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=5,
                dilation=5,
            ),
            nn.BatchNorm1d(int(512 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_d4 = Residual(
            int((512 + 1024) * hidden_dim_scaling + 0.5),
            int(512 * hidden_dim_scaling + 0.5),
        )

        self.decoder_l5 = nn.Sequential(
            nn.Conv1d(
                int((1024 + 2048) * hidden_dim_scaling + 0.5),
                int(1024 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(int(1024 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
            nn.Conv1d(
                int(1024 * hidden_dim_scaling + 0.5),
                int(1024 * hidden_dim_scaling + 0.5),
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm1d(int(1024 * hidden_dim_scaling + 0.5)),
            nn.ReLU(),
        )
        self.residual_d5 = Residual(
            int((1024 + 2048) * hidden_dim_scaling + 0.5),
            int(1024 * hidden_dim_scaling + 0.5),
        )

        self.init_type = init_type
        self.apply(self.init_weights)

    def init_weights(self, m):
        init_type = self.init_type
        if isinstance(m, nn.Conv1d):
            if init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.residual_e1(x, self.encoder_l1(x))

        x = rearrange(x1, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x2 = self.residual_e2(x, self.encoder_l2(x))

        x = rearrange(x2, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x3 = self.residual_e3(x, self.encoder_l3(x))

        x = rearrange(x3, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x4 = self.residual_e4(x, self.encoder_l4(x))

        x = rearrange(x4, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x5 = self.residual_e5(x, self.encoder_l5(x))

        # Bottleneck
        x = rearrange(x5, "B C (T1 T2) -> B (C T2) T1", T2=2)
        x = self.residual_bottleneck(x, self.bottleneck(x))

        # Decoder
        x = self.decoder_upsample(x)
        x = torch.cat((x, x5), dim=1)
        x = self.residual_d5(x, self.decoder_l5(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x4), dim=1)
        x = self.residual_d4(x, self.decoder_l4(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x3), dim=1)
        x = self.residual_d3(x, self.decoder_l3(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x2), dim=1)
        x = self.residual_d2(x, self.decoder_l2(x))

        x = self.decoder_upsample(x)
        x = torch.cat((x, x1), dim=1)
        x = self.residual_d1(x, self.decoder_l1(x))

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, init_type="xavier_normal"):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            ),
        )
        self.init_type = init_type
        self.apply(self.init_weights)

    def init_weights(self, m):
        init_type = self.init_type
        if isinstance(m, nn.Conv1d):
            if init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer(x)


class FullModel(nn.Module):
    def __init__(self, in_channels, out_channels, init_type="xavier_normal"):
        super(FullModel, self).__init__()
        self.init_type = init_type
        self.encoder = UNetEncoder(in_channels=in_channels, out_channels=128)
        self.seg_decoder = Decoder(in_channels=128, out_channels=out_channels)
        self.edge_decoder = nn.Sequential(
            Decoder(in_channels=128, out_channels=2),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        init_type = self.init_type
        if isinstance(m, nn.Conv1d):
            if init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.encoder(x)
        seg = self.seg_decoder(h)
        edge = self.edge_decoder(h)
        ctc = torch.cat((edge[:, [1], :], seg[:, 1:, :]), dim=1)
        return h, seg, ctc, edge


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, decay=None):
        if decay is not None:
            self.decay = decay
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
