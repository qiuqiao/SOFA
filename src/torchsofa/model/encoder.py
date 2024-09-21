from torch import nn

from ..network.layer.conv_stft import ConvMelSpectrogram


class Encoder(nn.Module):
    def __init__(
        self,
        network,
        mel_args: dict,
    ):
        super().__init__()
        self.stft = ConvMelSpectrogram(**mel_args)
        self.network = network

    def forward(self, wav):
        return self.network(self.stft(wav))
