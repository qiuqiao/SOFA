import click
from train import LitForcedAlignmentModel
import pathlib
import torch
# import textgrid
import lightning as pl
import modules.g2p


@click.command()
@click.option('--ckpt', '-c',
              default='ckpt/mandarin_opencpop-extension_singing/lightning_logs/version_0/checkpoints/epoch=0-step=500.ckpt',
              type=str, help='path to the checkpoint')
@click.option('--input', '-i', default='segments', type=str, help='path to the input folder')
@click.option('--output', '-o', default='segments', type=str, help='path to the output folder')
@click.option("--mode", "-m", default="force", type=click.Choice(["force", "match"]))  # TODO: add asr mode
@click.option('--g2p', '-g', default='Dictionary', type=str, help='name of the g2p class')
@click.option('--dictionary', '-d', default='dictionary/opencpop-extension.txt', type=str,
              help='path to the dictionary')
def main(ckpt, input, output, mode, g2p, **g2p_kwargs):
    if not g2p.endswith('G2P'):
        g2p += 'G2P'
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**g2p_kwargs)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(input).rglob('*.wav'))
    torch.set_grad_enabled(False)
    model = LitForcedAlignmentModel.load_from_checkpoint(ckpt)
    # model.set_infer_params(kwargs)
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)
    # print(predictions)
    # save_textgrids(output, predictions, word_seq_list, ph_idx_to_word_idx_list)
    # save_htk(output, predictions, word_seq_list, ph_idx_to_word_idx_list)
    # save_transcriptions(output, predictions, word_seq_list, ph_idx_to_word_idx_list)


if __name__ == "__main__":
    main()
