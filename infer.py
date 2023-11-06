import click
from train import LitForcedAlignmentModel
import pathlib
import torch
import textgrid
from modules.utils.load_wav import load_wav
from modules.utils.get_melspec import MelSpecExtractor


class ForcedAlignmentModelInferer:
    def __init__(self, ckpt_path: str, device):
        self.model = LitForcedAlignmentModel.load_from_checkpoint(ckpt_path)
        self.model.eval()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model.to(self.device)

        self.sample_rate = self.model.hparams.melspec_config["sample_rate"]
        self.get_melspec = MelSpecExtractor(**self.model.hparams.melspec_config, device=self.device)

    def infer(self, input_folder: str, output_folder: str):
        # load dataset list
        wav_path_list = pathlib.Path(input_folder).rglob('*.wav')
        for wav_path in wav_path_list:
            tg = self.infer_once(wav_path)
            if tg is None:
                continue
            # save textgrid to output folder

    def infer_once(self, wav_path):
        lab_path = wav_path.parent / f"{wav_path.stem}.lab"
        if not lab_path.exists():
            return None
        # forward
        waveform = load_wav(wav_path, self.device, self.sample_rate)
        melspec = self.get_melspec(waveform).unsqueeze(0)
        padding_length = 32 - (melspec.shape[-1] % 32)
        (
            ph_frame_pred,  # (B, T, vocab_size)
            ph_edge_pred,  # (B, T, 2)
            ctc_pred,  # (B, T, vocab_size)
        ) = self.model(melspec)
        print(ph_frame_pred.shape, ph_edge_pred.shape, ctc_pred.shape)
        # decode


@click.command()
@click.option('--ckpt', '-c',
              default='ckpt/mandarin_opencpop-extension_singing/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt',
              type=str, help='path to the checkpoint')
@click.option('--input', '-i', default='segments', type=str, help='path to the input folder')
@click.option('--output', '-o', default='segments', type=str, help='path to the output folder')
@click.option('--device', '-d', default=None, type=str, help='device to use')
def main(ckpt, input, output, device):
    inferer = ForcedAlignmentModelInferer(ckpt, device)
    inferer.infer(input, output)


if __name__ == "__main__":
    main()
