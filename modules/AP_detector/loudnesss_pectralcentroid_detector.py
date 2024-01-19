import numpy as np
import torch
import torch.nn as nn

from modules.AP_detector.base_detector import BaseAPDetector
from modules.utils.load_wav import load_wav


class LoudnessSpectralcentroidAPDetector(BaseAPDetector):
    def __init__(self, **kwargs):
        self.spectral_centroid_threshold = 40
        self.spl_threshold = 20

        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

        self.sample_rate = 44100
        self.hop_length = 512
        self.n_fft = 2048
        self.win_length = 1024
        self.hann_window = torch.hann_window(self.win_length).to(self.device)

        self.conv = nn.Conv1d(
            1, 1, self.hop_length, self.hop_length, self.hop_length // 2, bias=False
        ).to(self.device)
        self.conv.requires_grad_(False)
        self.conv.weight.data.fill_(1.0 / self.hop_length)

    def _get_spl(self, wav):
        out = self.conv(wav.pow(2).unsqueeze(0).unsqueeze(0))
        out = 20 * torch.log10(out.sqrt() / 2 * 10e5)
        return out.squeeze(0).squeeze(0)

    def _get_spectral_centroid(self, wav):
        wav = nn.functional.pad(wav, (self.n_fft // 2, (self.n_fft + 1) // 2))
        fft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=False,
            return_complex=True,
        )
        magnitude = fft.abs().pow(2)
        magnitude = magnitude / magnitude.sum(dim=-2, keepdim=True)

        spectral_centroid = torch.sum(
            (1 + torch.arange(0, self.n_fft // 2 + 1))
            .float()
            .unsqueeze(-1)
            .to(self.device)
            * magnitude,
            dim=0,
        )

        return spectral_centroid

    def _get_diff_intervals(self, intervals_a, intervals_b):
        # get complement of interval_b
        if intervals_a.shape[0] <= 0:
            return np.array([])
        if intervals_b.shape[0] <= 0:
            return intervals_a
        intervals_b = np.stack(
            [
                np.concatenate([[0.0], intervals_b[:, 1]]),
                np.concatenate([intervals_b[:, 0], intervals_a[[-1], [-1]]]),
            ],
            axis=-1,
        )
        intervals_b = intervals_b[(intervals_b[:, 0] < intervals_b[:, 1]), :]

        idx_a = 0
        idx_b = 0
        intersection_intervals = []
        while idx_a < intervals_a.shape[0] and idx_b < intervals_b.shape[0]:
            start_a, end_a = intervals_a[idx_a]
            start_b, end_b = intervals_b[idx_b]
            if end_a <= start_b:
                idx_a += 1
                continue
            if end_b <= start_a:
                idx_b += 1
                continue
            intersection_intervals.append([max(start_a, start_b), min(end_a, end_b)])
            if end_a < end_b:
                idx_a += 1
            else:
                idx_b += 1

        return np.array(intersection_intervals)

    def _process_one(
        self,
        wav_path,
        wav_length,
        confidence,
        ph_seq,
        ph_intervals,
        word_seq,
        word_intervals,
    ):
        # input:
        #     wav_path: pathlib.Path
        #     ph_seq: list of phonemes, SP is the silence phoneme.
        #     ph_intervals: np.ndarray of shape (n_ph, 2), ph_intervals[i] = [start, end]
        #                   means the i-th phoneme starts at start and ends at end.
        #     word_seq: list of words.
        #     word_intervals: np.ndarray of shape (n_word, 2), word_intervals[i] = [start, end]

        # output: same as the input.
        wav = load_wav(wav_path, self.device, self.sample_rate)
        wav = 0.01 * (wav - wav.mean()) / wav.std()

        # ap intervals
        spl = self._get_spl(wav)
        spectral_centroid = self._get_spectral_centroid(wav)
        ap_frame = (spl > self.spl_threshold) & (
            spectral_centroid > self.spectral_centroid_threshold
        )
        ap_frame_diff = torch.diff(
            torch.cat(
                [
                    torch.tensor([0], device=self.device),
                    ap_frame,
                    torch.tensor([0], device=self.device),
                ]
            ),
            dim=0,
        )
        ap_start_idx = torch.where(ap_frame_diff == 1)[0]
        ap_end_idx = torch.where(ap_frame_diff == -1)[0]
        ap_intervals = torch.stack([ap_start_idx, ap_end_idx], dim=-1) * (
            self.hop_length / self.sample_rate
        )
        ap_intervals = self._get_diff_intervals(
            ap_intervals.cpu().numpy(), word_intervals
        )
        if ap_intervals.shape[0] <= 0:
            return (
                wav_path,
                wav_length,
                confidence,
                ph_seq,
                ph_intervals,
                word_seq,
                word_intervals,
            )
        ap_intervals = ap_intervals[(ap_intervals[:, 1] - ap_intervals[:, 0]) > 0.1, :]

        # merge
        ap_tuple_list = [
            ("AP", ap_start, ap_end)
            for (ap_start, ap_end) in zip(ap_intervals[:, 0], ap_intervals[:, 1])
        ]
        word_tuple_list = [
            (word, word_start, word_end)
            for (word, (word_start, word_end)) in zip(word_seq, word_intervals)
        ]
        word_tuple_list.extend(ap_tuple_list)
        ph_tuple_list = [
            (ph, ph_start, ph_end)
            for (ph, (ph_start, ph_end)) in zip(ph_seq, ph_intervals)
        ]
        ph_tuple_list.extend(ap_tuple_list)

        # sort
        word_tuple_list.sort(key=lambda x: x[1])
        ph_tuple_list.sort(key=lambda x: x[1])

        ph_seq = [ph for (ph, _, _) in ph_tuple_list]
        ph_intervals = np.array([(start, end) for (_, start, end) in ph_tuple_list])

        word_seq = [word for (word, _, _) in word_tuple_list]
        word_intervals = np.array([(start, end) for (_, start, end) in word_tuple_list])

        return (
            wav_path,
            wav_length,
            confidence,
            ph_seq,
            ph_intervals,
            word_seq,
            word_intervals,
        )
