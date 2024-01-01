import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling


class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=64,
        num_embeddings=128,
        block=ResidualBasicBlock,
        down_sampling=DownSampling,
        up_sampling=UpSampling,
        hidden_dims=128,
        down_sampling_factor=2,
        down_sampling_times=3,
        channels_scaleup_factor=1.5,
    ):
        super(PhonemeEncoder, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            num_embeddings,
        )
        self.encoder = UNetBackbone(
            num_embeddings,
            num_embeddings,
            block=block,
            down_sampling=down_sampling,
            up_sampling=up_sampling,
            hidden_dims=hidden_dims,
            down_sampling_factor=down_sampling_factor,
            down_sampling_times=down_sampling_times,
            channels_scaleup_factor=channels_scaleup_factor,
        )

    def forward(self, ph_seq):
        embedding = self.embedding(ph_seq)
        return embedding, self.encoder(embedding)  # (B S E), (B S E)


class ForcedAlignmentModel(nn.Module):
    def __init__(
        self,
        n_mels=128,
        vocab_size=64,
        num_embeddings=128,
        audio_encoder={
            "hidden_dims": 128,
            "down_sampling_factor": 3,
            "down_sampling_times": 7,
            "channels_scaleup_factor": 1.5,
        },
        phoneme_encoder={
            "hidden_dims": 128,
            "down_sampling_factor": 2,
            "down_sampling_times": 3,
            "channels_scaleup_factor": 1.5,
        },
    ):
        super(ForcedAlignmentModel, self).__init__()
        self.num_embeddings = num_embeddings

        self.audio_encoder = UNetBackbone(
            n_mels,
            num_embeddings,
            block=ResidualBasicBlock,
            down_sampling=DownSampling,
            up_sampling=UpSampling,
            **audio_encoder,
        )
        self.phoneme_encoder = PhonemeEncoder(
            vocab_size,
            num_embeddings,
            block=ResidualBasicBlock,
            down_sampling=DownSampling,
            up_sampling=UpSampling,
            **phoneme_encoder,
        )
        # self.key_shift = nn.Parameter(torch.ones(2) * -1)
        # self.prior_weight = nn.Parameter(torch.ones(1) * 0.1)
        # self._cache = {}

    def forward(self, input_feature, ph_seq):
        # (B, T, E)
        audio_encoded = self.audio_encoder(input_feature)
        audio_encoded = audio_encoded / audio_encoded.norm(dim=-1, keepdim=True)
        # (B S E)
        phoneme_embed, phoneme_encoded = self.phoneme_encoder(ph_seq)

        # (B T S)
        attn_logits = (
            torch.matmul(audio_encoded, phoneme_encoded.transpose(1, 2))
            / self.num_embeddings
        )
        # TODO:add dummy_class
        # TODO: add prior knowledge TODO:如果add了prior就不需要mask，prior里带了
        return attn_logits


if __name__ == "__main__":
    B, T, S, E, n_mels = 4, 100, 10, 128, 128
    ph_seq = torch.randint(0, 64, (B, S))
    melspec = torch.randn(B, T, n_mels)

    model = ForcedAlignmentModel()

    attn_log_probs = model(melspec, ph_seq)
    print(attn_log_probs.shape)
    plt.imshow(
        attn_log_probs[0].exp().T.detach().numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.show()
