import modules.rmvpe

melspec_transform = None


class MelSpecExtractor:
    def __init__(self, n_mels, sample_rate, win_length, hop_length, fmin, fmax, device):
        global melspec_transform
        if melspec_transform is None:
            melspec_transform = modules.rmvpe.MelSpectrogram(
                n_mel_channels=n_mels,
                sampling_rate=sample_rate,
                win_length=win_length,
                hop_length=hop_length,
                mel_fmin=fmin,
                mel_fmax=fmax,
            ).to(device)

    def __call__(self, waveform):
        return melspec_transform(waveform.unsqueeze(0)).squeeze(0)
