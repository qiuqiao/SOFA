import torch
from torchfcpe import spawn_bundled_infer_model

import modules.rmvpe
from modules.utils.load_wav import load_wav

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


class RMSExtractor:
    def __init__(
        self,
        hop_length=512,
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conv = torch.nn.Conv1d(
            1, 1, hop_length * 2, stride=hop_length, bias=False
        ).to(device)
        self.conv.weight.data = torch.ones_like(self.conv.weight.data).to(device)
        self.conv.weight.data = self.conv.weight.data / torch.sum(self.conv.weight.data)
        self.conv.requires_grad_(False)

    def __call__(self, waveform):
        """_summary_

        Args:
            waveform (torch.tensor): (T,)
        """
        squared_waveform = waveform**2
        conv_mean = self.conv(squared_waveform.unsqueeze(0))
        rms = torch.sqrt(conv_mean)[0]

        return rms


class FeatureExtractor:

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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.get_melspec = MelSpecExtractor(
            n_mels,
            sample_rate,
            win_length,
            hop_length,
            n_fft,
            fmin,
            fmax,
            clamp,
            device=None,
        )

        self.fcpe = spawn_bundled_infer_model(device=device)

        self.get_rms = RMSExtractor(hop_length=hop_length, device=device)

        self.sample_rate = sample_rate

    def __call__(self, waveform, key_shifts=[0]):
        melspecs = torch.stack(
            [self.get_melspec(waveform, key_shift=i) for i in key_shifts], dim=0
        )
        melspecs = (
            melspecs - melspecs.mean(dim=[-1, -2], keepdim=True)
        ) / melspecs.std(dim=[-1, -2], keepdim=True)
        T = melspecs.shape[-1]

        f0, uv = self.fcpe.infer(
            waveform.unsqueeze(0).unsqueeze(-1),
            sr=self.sample_rate,
            interp_uv=True,
            output_interp_target_length=T,
            retur_uv=True,  # TODO: change to return_uv after updating torchfcpe
        )
        note = torch.log2(f0 / 440) * 12 + 69
        diff_note = torch.diff(
            note, 1, prepend=torch.zeros(1, 1, 1).to(self.device), dim=1
        )
        diff_note = diff_note * (diff_note.abs() < 6)

        diff_notes = torch.cat([diff_note for _ in key_shifts], dim=0).transpose(1, 2)
        uvs = torch.cat([uv for _ in key_shifts], dim=0).transpose(1, 2)

        rms = self.get_rms(waveform)
        if len(rms) < T:
            rms = torch.nn.functional.pad(rms, (0, T - len(rms)))
        elif len(rms) > T:
            rms = rms[:T]
        rms = rms.unsqueeze(0).unsqueeze(0).expand(len(key_shifts), -1, -1)

        features = torch.cat([melspecs, diff_notes, uvs, rms], dim=1)
        return features

        # print(melspecs.shape, diff_notes.shape, uvs.shape)
        # plt.plot(diff_notes[0].squeeze().cpu())
        # plt.show()


if __name__ == "__main__":
    extractor = FeatureExtractor(128, 44100, 1024, 512, 2048, 40, 16000, 0.00001)
    # extractor = RMSExtractor()
    waveform = load_wav(
        "test.wav",
        "cuda",
        44100,
    )
    # print(waveform.shape)
    features = extractor(waveform, [0, 7])
    print(features.shape)
