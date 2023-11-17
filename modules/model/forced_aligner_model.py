import torch
import torch.nn as nn

from modules.layer.backbone.Unet import UNetBackbone
from modules.layer.block.conformer import ForwardBackwardConformerBlock


class ForcedAlignmentModel(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims_ph_frame,
        hidden_dims=64,
        init_type="xavier_uniform",
        **kwargs
    ):
        super(ForcedAlignmentModel, self).__init__()

        self.init_type = init_type
        self.backbone = UNetBackbone(
            input_dims,
            hidden_dims,
            hidden_dims,
            block=ForwardBackwardConformerBlock,
            **kwargs
        )
        self.ph_frame_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Hardswish(),
            nn.Linear(hidden_dims, output_dims_ph_frame),
        )
        self.ph_edge_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Hardswish(),
            nn.Linear(hidden_dims, 2),
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        init_type = self.init_type
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            if init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        h = self.backbone(x)
        ph_frame = self.ph_frame_head(h)
        ph_edge = self.ph_edge_head(h)
        ctc = torch.cat((ph_edge[:, :, [0]], ph_frame[:, :, 1:]), dim=-1)
        return ph_frame, ph_edge, ctc


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
