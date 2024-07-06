import json
import pathlib
import warnings
from typing import Dict, List, Set, Tuple

import click
import textgrid
import tqdm


class Interval:
    def __init__(self, mark: str, start: float, end: float):
        self.mark = mark
        self.start = start
        self.end = end

    def __str__(self):
        return f"{self.__class__.__name__}(mark={self.mark}, start={self.start}, end={self.end})"


class Boundary:
    def __init__(self, mark: str, position: float):
        self.mark = mark
        self.position = position

    def __str__(self):
        return f"{self.__class__.__name__}(mark={self.mark}, position={self.position})"


def intervals_to_boundaries(intervals: List[Interval]) -> List[Boundary]:
    """
    Convert intervals to boundaries.
    """
    if len(intervals) == 0:
        return []
    boundaries = [Boundary(mark=None, position=0.0)]
    for interval in intervals:
        if boundaries[-1].mark is None and boundaries[-1].position == interval.start:
            boundaries[-1].mark = interval.mark
        else:
            boundaries.append(Boundary(mark=interval.mark, position=interval.start))
        boundaries.append(Boundary(mark=None, position=interval.end))
    return boundaries


def match_boundaries(
    boundaries1: List[Boundary], boundaries2: List[Boundary]
) -> Set[Tuple[Boundary, Boundary]]:
    """
    Match the beginnings and endings of non-space intervals between the given boundaries.
    """
    mappings: Set[Tuple[Boundary, Boundary]] = set()
    if len(boundaries1) == 0 or len(boundaries2) == 0:
        return mappings
    i = j = 0
    num_iters = min(
        len([b for b in boundaries1 if b.mark is not None]),
        len([b for b in boundaries2 if b.mark is not None]),
    )
    for _ in range(num_iters):
        # find beginning boundaries
        while boundaries1[i].mark is None:
            i += 1
        while boundaries2[j].mark is None:
            j += 1
        mappings.add((boundaries1[i], boundaries2[j]))
        # find ending boundaries
        i += 1
        j += 1
        if i < len(boundaries1) and j < len(boundaries2):
            mappings.add((boundaries1[i], boundaries2[j]))
    return mappings


class Metric:
    """
    A torchmetrics.Metric-like class with similar methods but lowered computing overhead.
    """

    def update(self, pred, target):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class BoundaryEditDistance(Metric):
    """
    The total moving distance from the predicted boundaries to the target boundaries.
    """

    def __init__(self):
        self.distance = 0.0

    def update(self, pred: List[Interval], target: List[Interval]):
        assert len(pred) == len(
            target
        ), f"Number of intervals should be equal in pred and target ({len(pred)} != {len(target)})."
        if len(target) == 0:
            return
        # get boundaries of pred and target
        p_boundaries = intervals_to_boundaries(pred)
        t_boundaries = intervals_to_boundaries(target)

        # find boundary mappings
        mappings = match_boundaries(p_boundaries, t_boundaries)

        # compute the distance
        distance = sum(abs(b1.position - b2.position) for b1, b2 in mappings)
        self.distance += distance

    def compute(self):
        return self.distance

    def reset(self):
        self.distance = 0.0


class BoundaryEditRatio(Metric):
    """
    The boundary edit distance divided by the total duration of target intervals.
    """

    def __init__(self):
        self.distance_metric = BoundaryEditDistance()
        self.duration = 0.0

    def update(self, pred: List[Interval], target: List[Interval]):
        self.distance_metric.update(pred=pred, target=target)
        if len(target) > 0:
            self.duration += target[-1].end

    def compute(self):
        return self.distance_metric.compute() / self.duration

    def reset(self):
        self.distance_metric.reset()
        self.duration = 0.0


class BoundaryErrorRate(Metric):
    """
    The proportion of misplaced boundaries to all target boundaries under a given tolerance of distance.
    """

    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance
        self.errors = 0
        self.total = 0

    def update(self, pred: List[Interval], target: List[Interval]):
        if len(target) == 0:
            return
        # get boundaries of pred and target
        p_boundaries = intervals_to_boundaries(pred)
        t_boundaries = intervals_to_boundaries(target)

        # find boundary mappings
        mappings = match_boundaries(p_boundaries, t_boundaries)
        errors = sum(
            abs(b1.position - b2.position) > self.tolerance for b1, b2 in mappings
        )
        self.errors += errors
        self.total += len(t_boundaries)

    def compute(self):
        return self.errors / self.total

    def reset(self):
        self.errors = 0
        self.total = 0


def parse_intervals_from_textgrid(file: pathlib.Path) -> List[Interval]:
    """
    Parse the intervals from a Textgrid file and return interval tuples (mark, start, end).
    """
    tg = textgrid.TextGrid()
    tg.read(file, encoding="utf8")
    tier: textgrid.IntervalTier = None
    for t in tg.tiers:
        if isinstance(t, textgrid.IntervalTier) and t.name == "phones":
            tier = t
            break
    assert tier is not None, f'There are no phones tier in file "{file}".'
    return [Interval(mark=i.mark, start=i.minTime, end=i.maxTime) for i in tier]


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
    default="AP,SP,<AP>,<SP>,,pau,cl",
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
        "BoundaryErrorRate10ms": BoundaryErrorRate(tolerance=0.01),
        "BoundaryErrorRate20ms": BoundaryErrorRate(tolerance=0.02),
        "BoundaryErrorRate50ms": BoundaryErrorRate(tolerance=0.05),
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
        pred_intervals = parse_intervals_from_textgrid(pred_file)
        target_intervals = parse_intervals_from_textgrid(target_file)
        pred_intervals = [p for p in pred_intervals if p.mark not in ignored]
        target_intervals = [p for p in target_intervals if p.mark not in ignored]
        pred_interval_marks = [p.mark for p in pred_intervals]
        target_interval_marks = [p.mark for p in target_intervals]
        if pred_interval_marks != target_interval_marks:
            if strict:
                raise RuntimeError(
                    f"Phone sequences from prediction file and target file are not identical: "
                    f"{pred_interval_marks} in {pred_file} compared to {target_interval_marks} in {target_file}."
                )
            else:
                continue
        for metric in metrics.values():
            metric.update(pred_intervals, target_intervals)
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
