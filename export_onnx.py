import os.path
import pathlib

import click
import onnx
import onnxsim

from typing import Any

import lightning as pl
import yaml
from einops import repeat
from modules.layer.backbone.unet import UNetBackbone
from modules.layer.block.resnet_block import ResidualBasicBlock
from modules.layer.scaling.stride_conv import DownSampling, UpSampling

import torch
import torch.nn as nn
from librosa.filters import mel


class MelSpectrogram_ONNX(nn.Module):
    def __init__(
            self,
            n_mel_channels,
            sampling_rate,
            win_length,
            hop_length,
            n_fft=None,
            mel_fmin=0,
            mel_fmax=None,
            clamp=1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, center=True):
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=audio.device),
            center=center,
            return_complex=False
        )
        magnitude = torch.sqrt(torch.sum(fft ** 2, dim=-1))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class LitForcedAlignmentOnnx(pl.LightningModule):
    def __init__(
            self,
            vocab_text,
            model_config,
            melspec_config
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = yaml.safe_load(vocab_text)
        self.melspec_config = melspec_config

        self.backbone = UNetBackbone(
            melspec_config["n_mels"],
            model_config["hidden_dims"],
            model_config["hidden_dims"],
            ResidualBasicBlock,
            DownSampling,
            UpSampling,
            down_sampling_factor=model_config["down_sampling_factor"],  # 3
            down_sampling_times=model_config["down_sampling_times"],  # 7
            channels_scaleup_factor=model_config["channels_scaleup_factor"],  # 1.5
        )
        self.head = nn.Linear(
            model_config["hidden_dims"], self.vocab["<vocab_size>"] + 2
        )
        self.mel_extractor = MelSpectrogram_ONNX(
            melspec_config["n_mels"], melspec_config["sample_rate"], melspec_config["win_length"],
            melspec_config["hop_length"], melspec_config["n_fft"], melspec_config["fmin"], melspec_config["fmax"]
        )

    def forward(self, waveform, num_frames, ph_seq_id) -> Any:
        melspec = self.mel_extractor(waveform).detach()
        melspec = (melspec - melspec.mean()) / melspec.std()
        melspec = repeat(
            melspec, "B C T -> B C (T N)", N=self.melspec_config["scale_factor"]
        )

        h = self.backbone(melspec.transpose(1, 2))
        logits = self.head(h)
        ph_frame_logits = logits[:, :, 2:]
        ph_edge_logits = logits[:, :, 0]
        ctc_logits = torch.cat([logits[:, :, [1]], logits[:, :, 3:]], dim=-1)

        ph_mask = torch.zeros(self.vocab["<vocab_size>"])
        ph_mask[ph_seq_id] = 1
        ph_mask[0] = 1

        ph_frame_logits = ph_frame_logits[:, :num_frames, :]
        ph_edge_logits = ph_edge_logits[:, :num_frames]
        ctc_logits = ctc_logits[:, :num_frames, :]

        ph_mask = ph_mask.to(ph_frame_logits.device).unsqueeze(0).unsqueeze(0).logical_not() * 1e9
        ph_frame_pred = torch.nn.functional.softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1).squeeze(0)
        ph_prob_log = torch.log_softmax(ph_frame_logits.float() - ph_mask.float(), dim=-1).squeeze(0)
        ph_edge_pred = ((torch.nn.functional.sigmoid(ph_edge_logits.float()) - 0.1) / 0.8).clamp(0.0, 1.0)
        ph_edge_pred = ph_edge_pred.squeeze(0)
        ctc_logits = ctc_logits.float().squeeze(0)  # (ctc_logits.squeeze(0) - ph_mask)

        T, vocab_size = ph_frame_pred.shape

        # decode
        diff_ph_edge_pred = ph_edge_pred[1:] - ph_edge_pred[:-1]
        edge_diff = torch.cat((diff_ph_edge_pred, torch.tensor([0.0], device=ph_edge_pred.device)), dim=0)
        edge_prob = (ph_edge_pred + torch.cat(
            (torch.tensor([0.0], device=ph_edge_pred.device), ph_edge_pred[:-1]))).clamp(0, 1)
        return edge_diff, edge_prob, ph_prob_log, ctc_logits, T


@torch.no_grad()
@click.command(help='')
@click.option('--ckpt_path', required=True, metavar='DIR', help='Path to the checkpoint')
@click.option('--onnx_path', required=True, metavar='DIR', help='Path to the onnx')
def export(ckpt_path, onnx_path):
    assert ckpt_path is not None, "Checkpoint directory (ckpt_dir) cannot be None"

    assert os.path.exists(ckpt_path), f"Checkpoint path does not exist: {ckpt_path}"

    os.makedirs(pathlib.Path(onnx_path).parent, exist_ok=True)

    output_config = pathlib.Path(onnx_path).with_name('config.yaml')
    assert not os.path.exists(onnx_path), f"Error: The file '{onnx_path}' already exists."
    assert not output_config.exists(), f"Error: The file '{output_config}' already exists."

    model = LitForcedAlignmentOnnx.load_from_checkpoint(ckpt_path, strict=False)

    waveform = torch.randn((1, 44100), dtype=torch.float32)
    ph_seq_id = torch.zeros((1, 37), dtype=torch.int64)
    num_frames = torch.tensor(500, dtype=torch.int64)

    if torch.cuda.is_available():
        model.cuda()
        waveform = waveform.cuda()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (waveform, num_frames, ph_seq_id),
            onnx_path,
            input_names=['waveform', 'num_frames', 'ph_seq_id'],
            output_names=['edge_diff', 'edge_prob', 'ph_prob_log', 'ctc_logits', 'T'],
            dynamic_axes={
                'waveform': {1: 'n_samples'},
                'ph_seq_id': {1: 'n_samples'},
                'edge_diff': {1: 'n_samples'},
                'edge_prob': {1: 'n_samples'},
                'ph_prob_log': {1: 'n_samples'},
                'ctc_logits': {1: 'n_samples'}
            },
            opset_version=17
        )
        onnx_model, check = onnxsim.simplify(onnx_path, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(onnx_model, onnx_path)
        print(f'Model saved to: {onnx_path}')

    out_config = {'melspec_config': model.hparams.melspec_config,
                  'model_config': model.hparams.model_config,
                  'vocab': yaml.safe_load(model.hparams.vocab_text)
                  }
    with open(output_config, 'w') as file:
        yaml.dump(out_config, file, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    export()
