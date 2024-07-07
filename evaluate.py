import json
import pathlib
import warnings
from typing import Dict

import click
import tqdm
from textgrid import PointTier

from modules.utils import label
from modules.utils.metrics import (
    BoundaryEditRatio,
    IntersectionOverUnion,
    Metric,
    VlabelerEditRatio,
)


def remove_ignored_phonemes(ignored_phonemes_list: str, point_tier: PointTier):
    res_tier = PointTier(name=point_tier.name)
    if point_tier[0].mark not in ignored_phonemes_list:
        res_tier.addPoint(point_tier[0])
    for i in range(len(point_tier) - 1):
        if (
            point_tier[i].mark in ignored_phonemes_list
            and point_tier[i + 1].mark in ignored_phonemes_list
        ):
            continue

        res_tier.addPoint(point_tier[i + 1])

    return res_tier


@click.command(
    help="Calculate metrics between the FA predictions and the targets (ground truth)."
)
@click.argument(
    "pred",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    metavar="PRED_DIR",
)
@click.argument(
    "target",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    metavar="TARGET_DIR",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Compare files in subdirectories recursively",
)
@click.option(
    "--strict", "-s", is_flag=True, help="Raise errors on mismatching phone sequences"
)
@click.option(
    "--ignore",
    type=str,
    default="",  # AP,SP,<AP>,<SP>,,pau,cl
    help="Ignored phone marks, split by commas",
    show_default=True,
)
def main(pred: str, target: str, recursive: bool, strict: bool, ignore: str):
    pred_dir = pathlib.Path(pred)
    target_dir = pathlib.Path(target)
    if recursive:
        iterable = pred_dir.rglob("*.TextGrid")
    else:
        iterable = pred_dir.glob("*.TextGrid")
    ignored = ignore.split(",")
    metrics: Dict[str, Metric] = {
        "BoundaryEditRatio": BoundaryEditRatio(),
        "VlabelerEditRatio10ms": VlabelerEditRatio(move_tolerance=0.01),
        "VlabelerEditRatio20ms": VlabelerEditRatio(move_tolerance=0.02),
        "VlabelerEditRatio50ms": VlabelerEditRatio(move_tolerance=0.05),
        "IntersectionOverUnion": IntersectionOverUnion(),
    }

    cnt = 0
    for pred_file in tqdm.tqdm(iterable):
        target_file = target_dir / pred_file.relative_to(pred_dir)
        if not target_file.exists():
            warnings.warn(
                f'The prediction file "{pred_file}" has no matching target file, '
                f'which should be "{target_file}".',
                category=UserWarning,
            )
            warnings.filterwarnings("default")
            continue

        pred_tier = label.textgrid_from_file(pred_file)[-1]
        target_tier = label.textgrid_from_file(target_file)[-1]
        pred_tier = remove_ignored_phonemes(ignored, pred_tier)
        target_tier = remove_ignored_phonemes(ignored, target_tier)

        for metric in metrics.values():
            try:
                metric.update(pred_tier, target_tier)
            except AssertionError as e:
                if not strict:
                    warnings.warn(
                        f"Failed to evaluate metric {metric.__class__.__name__} for file {pred_file}: {e}",
                        category=UserWarning,
                    )
                    warnings.filterwarnings("default")
                    continue
                else:
                    raise e

        cnt += 1

    if cnt == 0:
        raise RuntimeError(
            "Unable to compare any files in the given directories. "
            "Matching files should have same names and same relative paths, "
            "containing the same phone sequences except for spaces."
        )
    result = {key: metric.compute() for key, metric in metrics.items()}
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
