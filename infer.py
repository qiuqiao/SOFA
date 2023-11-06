import click
from train import LitForcedAlignmentModel
import pathlib
import torch
import textgrid
from modules.utils.load_wav import load_wav
from modules.utils.get_melspec import MelSpecExtractor


class ForcedAlignmentModelInferer:
    def __init__(self, ckpt_path: str):
        # load model
        self.model = LitForcedAlignmentModel.load_from_checkpoint(ckpt_path)
        self.model.eval()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_rate = self.model.hparams.sample_rate
        self.get_melspec = MelSpecExtractor(n_mels=self.model.hparams.n_mels,
                                            sample_rate=self.model.hparams.sample_rate,
                                            win_length=self.model.hparams.win_length,
                                            hop_length=self.model.hparams.hop_length,
                                            fmin=self.model.hparams.fmin,
                                            fmax=self.model.hparams.fmax,
                                            device=self.model.hparams.device,
                                            )

    def infer(self, input_folder: str, output_folder: str):
        # load dataset list
        wav_path_list = pathlib.Path(input_folder).rglob('*.wav')
        for wav_path in wav_path_list:
            tg = self.infer_once(wav_path)
            if tg is not None:
                pass
                # save textgrid to output folder

    def infer_once(self, wav_path):
        lab_path = wav_path.parent / f"{wav_path.stem}.lab"
        print(lab_path)
        if not lab_path.exists():
            return None
        # forward
        waveform = load_wav(wav_path, self.device, self.sample_rate)
        print(waveform.shape)
        # decode


@click.command()
@click.option('--ckpt', '-c',
              default='ckpt/mandarin_opencpop-extension_singing/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt',
              type=str, help='path to the checkpoint')
@click.option('--input', '-i', default='segments', type=str, help='path to the input folder')
@click.option('--output', '-o', default='segments', type=str, help='path to the output folder')
def main(ckpt, input, output):
    inferer = ForcedAlignmentModelInferer(ckpt)
    inferer.infer(input, output)


if __name__ == "__main__":
    main()
