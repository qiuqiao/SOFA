import torch
import torch.nn as nn
from modules.layer.backbone.Unet import UNetBackbone


class ForcedAlignmentModel(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims, init_type="xavier_uniform"):
        super(ForcedAlignmentModel, self).__init__()

        self.init_type = init_type
        self.backbone=UNetBackbone()
        self.ph_frame_head=
        self.ph_edge_head=

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
