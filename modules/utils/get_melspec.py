import torch

import modules.rmvpe

melspec_transform = None


class MelSpecExtractor:
    def __init__(
        self,
        n_mels,
        sample_rate,
        win_length,
        hop_length,
        n_fft,
        fmin,
        fmax,
        clamp,
        device=None,
        scale_factor=None,
    ):
        global melspec_transform
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if melspec_transform is None:
            melspec_transform = modules.rmvpe.MelSpectrogram(
                n_mel_channels=n_mels,
                sampling_rate=sample_rate,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                mel_fmin=fmin,
                mel_fmax=fmax,
                clamp=clamp,
            ).to(device)

    def __call__(self, waveform, key_shift=0):
        return melspec_transform(waveform.unsqueeze(0), key_shift).squeeze(0)
