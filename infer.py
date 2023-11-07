import click
from train import LitForcedAlignmentModel
import pathlib
import torch
# import textgrid
from modules.utils.load_wav import load_wav
from modules.utils.get_melspec import MelSpecExtractor
import numpy as np
from modules.utils.plot import plot_for_test
import lightning as pl


@click.command()
@click.option('--ckpt', '-c',
              default='ckpt/mandarin_opencpop-extension_singing/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt',
              type=str, help='path to the checkpoint')
@click.option('--input', '-i', default='segments', type=str, help='path to the input folder')
@click.option('--output', '-o', default='segments', type=str, help='path to the output folder')
@click.option('--dictionary', '-d', default='dictionary/opencpop-extension.txt', type=str,
              help='path to the dictionary')
@click.option('--phoneme', '-p', default=False, is_flag=True, help='use phoneme mode')
@click.option("--matching", "-m", default=False, is_flag=True, help="use lyric matching mode")
def main(ckpt, input, output, **kwargs):  # dictionary, phoneme, matching, device
    torch.set_grad_enabled(False)
    model = LitForcedAlignmentModel.load_from_checkpoint(ckpt)
    model.set_infer_params(kwargs)
    wav_path_array = np.array(list(pathlib.Path(input).rglob('*.wav')))
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=wav_path_array, return_predictions=True)
    # save_textgrids(predictions, output)
    # save_htk(predictions, output)
    # save_transcriptions(predictions, output)


if __name__ == "__main__":
    main()
