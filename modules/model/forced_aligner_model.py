import torch
import torch.nn as nn

from modules.layer.backbone.Unet import UNetBackbone
from modules.layer.block.conformer import ForwardBackwardConformerBlock


class ForcedAlignmentModel(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        init_type="xavier_uniform",
        pretrained_model=None,
        **kwargs,
    ):
        super(ForcedAlignmentModel, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        if pretrained_model is not None:
            self.hidden_dims = pretrained_model.hidden_dims
            if self.hidden_dims != hidden_dims:
                print(
                    f"hidden_dims not match: {self.hidden_dims} (pretrained) vs {hidden_dims} (input), use {self.hidden_dims}"
                )
        else:
            self.hidden_dims = hidden_dims
        self.init_type = init_type

        self.input_proj = nn.Linear(self.input_dims, self.hidden_dims)
        self.backbone = UNetBackbone(
            self.hidden_dims,
            self.hidden_dims,
            self.hidden_dims,
            block=ForwardBackwardConformerBlock,
            **kwargs,
        )
        self.ph_frame_head = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.LayerNorm(self.hidden_dims),
            nn.Hardswish(),
            nn.Linear(self.hidden_dims, self.output_dims),
        )
        self.ph_edge_head = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.LayerNorm(self.hidden_dims),
            nn.Hardswish(),
            nn.Linear(self.hidden_dims, 2),
        )

        self.apply(self.init_weights)

        if pretrained_model is not None:
            try:
                self.backbone.load_state_dict(pretrained_model.backbone.state_dict())
            except Exception as e:
                raise e("block type not match")

            try:
                self.input_proj.load_state_dict(
                    pretrained_model.input_proj.state_dict()
                )
            except Exception:
                print("input_dims not match, input_proj not loaded")

            try:
                self.ph_frame_head.load_state_dict(
                    pretrained_model.ph_frame_head.state_dict()
                )
                self.ph_edge_head.load_state_dict(
                    pretrained_model.ph_edge_head.state_dict()
                )
            except Exception:
                print(
                    "output_dims not match, ph_frame_head and ph_edge_head not loaded"
                )

    def init_weights(self, m):
        init_type = self.init_type
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            if init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight.data)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight.data)
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
