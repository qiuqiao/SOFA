# Conversion between various label formats (.csv, .lab, .textgrid, etc.) and a unified format.
# TextGrid and PointTier are used as the unified format for efficient calculation.
# Point.time indicates the start time of the phoneme, consistent with Vlabeler's behavior.

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import textgrid as tg


def durations_to_tier(
    marks: List,
    durarions: Union[List, np.ndarray],
    name="phones",
    start_time=0.0,
) -> tg.PointTier:
    assert len(marks) == len(durarions)

    durarions = np.insert(durarions, 0, start_time)
    times = np.cumsum(durarions)
    marks.append("")

    tier = tg.PointTier(name=name)
    for time, mark in zip(times, marks):
        tier.add(time, mark)

    return tier


def interval_tier_to_point_tier(tier: tg.IntervalTier) -> tg.PointTier:
    point_tier = tg.PointTier(name=tier.name)
    point_tier.add(0.0, "")
    for interval in tier:
        if point_tier[-1].mark == "" and point_tier[-1].time == interval.minTime:
            point_tier[-1].mark = interval.mark
        else:
            point_tier.add(interval.minTime, interval.mark)
        point_tier.add(interval.maxTime, "")

    return point_tier


def point_tier_to_interval_tier(tier: tg.PointTier) -> tg.IntervalTier:
    interval_tier = tg.IntervalTier(name=tier.name)
    for idx in range(len(tier) - 1):
        interval_tier.add(tier[idx].time, tier[idx + 1].time, tier[idx].mark)
    return interval_tier


def tier_from_htk(lab_path: str, tier_name="phones") -> tg.PointTier:
    """Read a htk label file (nnsvs format) and return a PointTier object."""
    tier = tg.IntervalTier(name=tier_name)

    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            start, end, mark = line.strip().split()
            tier.add(int(start) / 1e7, int(end) / 1e7, mark)

    return interval_tier_to_point_tier(tier)


def textgrid_from_file(textgrid_path: str) -> tg.TextGrid:
    """Read a TextGrid file and return a TextGrid object."""
    textgrid = tg.TextGrid()
    textgrid.read(textgrid_path, encoding="utf-8")
    for idx, tier in enumerate(textgrid):
        if isinstance(tier, tg.IntervalTier):
            textgrid.tiers[idx] = interval_tier_to_point_tier(tier)

    return textgrid


def textgrids_from_csv(csv_path: str) -> List[Tuple[str, tg.TextGrid]]:
    """Read a CSV file and return a list of (filename, TextGrid) tuples."""
    textgrids = []

    df = pd.read_csv(csv_path)
    df = df.loc[:, ["name", "ph_seq", "ph_dur"]]

    for _, row in df.iterrows():
        textgrid = tg.TextGrid()
        tier = durations_to_tier(
            row["ph_seq"].split(), list(map(float, row["ph_dur"].split()))
        )
        textgrid.append(tier)

        textgrids.append((row["name"], textgrid))

    return textgrids


def save_tier_to_htk(tier: tg.PointTier, lab_path: str) -> None:
    """Save a PointTier object to a htk label file."""
    with open(lab_path, "w", encoding="utf-8") as f:
        for i in range(len(tier) - 1):
            f.write(
                "{:.0f} {:.0f} {}\n".format(
                    tier[i].time * 1e7, tier[i + 1].time * 1e7, tier[i].mark
                )
            )


def save_textgrid(path: str, textgrid: tg.TextGrid) -> None:
    """Save a TextGrid object to a TextGrid file."""
    for i in range(len(textgrid)):
        if textgrid[i].maxTime is None:
            textgrid[i].maxTime = textgrid[i][-1].time
        if isinstance(textgrid[i], tg.PointTier):
            textgrid.tiers[i] = point_tier_to_interval_tier(textgrid[i])
    textgrid.write(path)


def save_textgrids_to_csv(
    path: str,
    textgrids: List[Tuple[str, tg.TextGrid]],
    precision=6,
) -> None:
    """Save a list of (filename, TextGrid) tuples to a CSV file."""
    rows = []
    for name, textgrid in textgrids:
        tier = textgrid[-1]
        ph_seq = " ".join(
            ["" if point.mark == "" else point.mark for point in tier[:-1]]
        )
        ph_dur = " ".join(
            [
                "{:.{}n}".format(ed.time - st.time, precision)
                for st, ed in zip(tier[:-1], tier[1:])
            ]
        )
        rows.append([name, ph_seq, ph_dur])

    df = pd.DataFrame(rows, columns=["name", "ph_seq", "ph_dur"])
    df.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    textgrid = textgrid_from_file("test/label/tg.TextGrid")
    save_textgrid("test/label/tg_out.TextGrid", textgrid)
    # # Save the TextGrid object to a htk label file
    # # save_htk(textgrid, "example_out.lab")
    # # Convert a TextGrid file to a TextGrid object
    # textgrid = from_textgrid("example.TextGrid")
    # # Save the TextGrid object to a TextGrid file
    # textgrid.write("example_out.TextGrid")
