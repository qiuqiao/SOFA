import pathlib

import click
import lightning as pl
import textgrid
import torch

import modules.AP_detector
import modules.g2p
from train import LitForcedAlignmentTask

MIN_SP_LENGTH = 0.1


def add_SP(word_seq, word_intervals, wav_length):
    word_seq_res = []
    word_intervals_res = []
    if len(word_seq) == 0:
        word_seq_res.append("SP")
        word_intervals_res.append([0, wav_length])
        return word_seq_res, word_intervals_res

    word_seq_res.append("SP")
    word_intervals_res.append([0, word_intervals[0, 0]])
    for word, (start, end) in zip(word_seq, word_intervals):
        if word_intervals_res[-1][1] < start:
            word_seq_res.append("SP")
            word_intervals_res.append([word_intervals_res[-1][1], start])
        word_seq_res.append(word)
        word_intervals_res.append([start, end])
    if word_intervals_res[-1][1] < wav_length:
        word_seq_res.append("SP")
        word_intervals_res.append([word_intervals_res[-1][1], wav_length])
    if word_intervals[0, 0] <= 0:
        word_seq_res = word_seq_res[1:]
        word_intervals_res = word_intervals_res[1:]

    return word_seq_res, word_intervals_res


def fill_small_gaps(word_seq, word_intervals):
    for idx in range(len(word_seq) - 1):
        if word_intervals[idx, 1] < word_intervals[idx + 1, 0]:
            if word_intervals[idx + 1, 0] - word_intervals[idx, 1] < MIN_SP_LENGTH:
                if word_seq[idx] == "AP":
                    word_intervals[idx, 1] = word_intervals[idx + 1, 0]
                elif word_seq[idx + 1] == "AP":
                    word_intervals[idx + 1, 0] = word_intervals[idx, 1]
                else:
                    mean = (word_intervals[idx, 1] + word_intervals[idx + 1, 0]) / 2
                    word_intervals[idx, 1] = mean
                    word_intervals[idx + 1, 0] = mean

    return word_seq, word_intervals


def post_processing(predictions):
    print("Post-processing...")

    res = []
    for (
        wav_path,
        wav_length,
        ph_seq,
        ph_intervals,
        word_seq,
        word_intervals,
    ) in predictions:
        try:
            # fill small gaps
            word_seq, word_intervals = fill_small_gaps(word_seq, word_intervals)
            ph_seq, ph_intervals = fill_small_gaps(ph_seq, ph_intervals)
            # add SP
            word_seq, word_intervals = add_SP(word_seq, word_intervals, wav_length)
            ph_seq, ph_intervals = add_SP(ph_seq, ph_intervals, wav_length)

            res.append(
                [wav_path, wav_length, ph_seq, ph_intervals, word_seq, word_intervals]
            )
        except Exception as e:
            e.args += (wav_path,)
            raise e
    return res


def save_textgrids(predictions):
    print("Saving TextGrids...")

    for (
        wav_path,
        wav_length,
        ph_seq,
        ph_intervals,
        word_seq,
        word_intervals,
    ) in predictions:
        tg = textgrid.TextGrid()
        word_tier = textgrid.IntervalTier(name="words")
        ph_tier = textgrid.IntervalTier(name="phones")

        for word, (start, end) in zip(word_seq, word_intervals):
            word_tier.add(start, end, word)

        for ph, (start, end) in zip(ph_seq, ph_intervals):
            ph_tier.add(minTime=float(start), maxTime=end, mark=ph)

        tg.append(word_tier)
        tg.append(ph_tier)

        label_path = (
            wav_path.parent / "TextGrid" / wav_path.with_suffix(".TextGrid").name
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        tg.write(label_path)


def save_htk(predictions):
    print("Saving htk labels...")

    for (
        wav_path,
        wav_length,
        ph_seq,
        ph_intervals,
        word_seq,
        word_intervals,
    ) in predictions:
        label = ""
        for ph, (start, end) in zip(ph_seq, ph_intervals):
            start_time = int(float(start) * 10000000)
            end_time = int(float(end) * 10000000)
            label += f"{start_time} {end_time} {ph}\n"
        label_path = (
            wav_path.parent / "htk" / "phones" / wav_path.with_suffix(".lab").name
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(label)
            f.close()

        label = ""
        for word, (start, end) in zip(word_seq, word_intervals):
            start_time = int(float(start) * 10000000)
            end_time = int(float(end) * 10000000)
            label += f"{start_time} {end_time} {word}\n"
        label_path = (
            wav_path.parent / "htk" / "words" / wav_path.with_suffix(".lab").name
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(label)
            f.close()


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
    default="LoudnessSpectralcentroidAPDetector",  # "NoneAPDetector",
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
@click.option(
    "--in_format",
    "-if",
    default="lab",
    required=False,
    type=str,
    help="File extension of input transcriptions. Default: lab",
)
@click.option(
    "--out_formats",
    "-of",
    default="TextGrid,htk",
    required=False,
    type=str,
    help="Types of output file, separated by comma. Supported types: TextGrid(textgrid,praat), htk(lab,nnsvs,sinsy)",
)
def main(ckpt, folder, mode, g2p, ap_detector, in_format, out_formats, **kwargs):
    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(modules.g2p, g2p)
    grapheme_to_phoneme = g2p_class(**kwargs)
    out_formats = [i.strip() for i in out_formats.split(",")]

    if not ap_detector.endswith("APDetector"):
        ap_detector += "APDetector"
    AP_detector_class = getattr(modules.AP_detector, ap_detector)
    get_AP = AP_detector_class(**kwargs)

    grapheme_to_phoneme.set_in_format(in_format)
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    torch.set_grad_enabled(False)
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt)
    model.set_inference_mode(mode)
    trainer = pl.Trainer(logger=False)
    predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)

    predictions = get_AP.process(predictions)
    predictions = post_processing(predictions)
    if "TextGrid" in out_formats or "textgrid" in out_formats or "praat" in out_formats:
        save_textgrids(predictions)
    if (
        "htk" in out_formats
        or "lab" in out_formats
        or "nnsvs" in out_formats
        or "sinsy" in out_formats
    ):
        save_htk(predictions)
    # save_transcriptions(output, predictions)
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == "__main__":
    main()
