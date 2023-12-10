import pathlib

import click
import lightning as pl
import textgrid
import torch

import modules.AP_detector
import modules.g2p
from train import LitForcedAlignmentTask


def save_textgrids(predictions):
    print("Saving TextGrids...")

    for wav_path, ph_seq, ph_intervals, word_seq, word_intervals in predictions:
        tg = textgrid.TextGrid()
        word_tier = textgrid.IntervalTier(name="words")
        ph_tier = textgrid.IntervalTier(name="phones")

        for word, (start, end) in zip(word_seq, word_intervals):
            if len(word_tier) > 0 and word_tier[-1].maxTime < start:
                word_tier.add(word_tier[-1].maxTime, start, "SP")
            word_tier.add(start, end, word)

        for ph, (start, end) in zip(ph_seq, ph_intervals):
            if len(ph_tier) > 0 and ph_tier[-1].maxTime < start:
                ph_tier.add(minTime=ph_tier[-1].maxTime, maxTime=start, mark="SP")
            ph_tier.add(minTime=start, maxTime=end, mark=ph)

        tg.append(word_tier)
        tg.append(ph_tier)
        tg.write(wav_path.with_suffix(".TextGrid"))


@click.command()
@click.option(
    "--ckpt",
    "-c",
    default=None,
    required=True,
    type=str,
    help="path to the checkpoint",
)
@click.option(
    "--folder", "-f", default="segments", type=str, help="path to the input folder"
)
@click.option(
    "--mode", "-m", default="force", type=click.Choice(["force", "match"])
)  # TODO: add asr mode
@click.option(
    "--g2p", "-g", default="Dictionary", type=str, help="name of the g2p class"
)
@click.option(
    "--ap_detector",
    "-a",
    default="NoneAPDetector",  # "LoudnessSpectralcentroidAPDetector",
    type=str,
    help="name of the AP detector class",
)
@click.option(
    "--dictionary",
    "-d",
    default="dictionary/opencpop-extension.txt",
    type=str,
    help="(only used when --g2p=='Dictionary') path to the dictionary",
)
def main(ckpt, folder, mode, g2p, ap_detector, **kwargs):
    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)

    if not ap_detector.endswith("APDetector"):
        ap_detector += "APDetector"
    AP_detector_class = getattr(modules.AP_detector, ap_detector)
    get_AP = AP_detector_class(**kwargs)

    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    model.set_inference_mode(mode)
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions = get_AP.process(predictions)

    save_textgrids(predictions)
    # save_htk(output, predictions)
    # save_transcriptions(output, predictions)
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()
