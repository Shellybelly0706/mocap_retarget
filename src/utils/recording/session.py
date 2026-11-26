"""Read the explicit frame mapping produced by timestamp preprocessing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.avatars.pose import load_bones
from utils.optitrack.csv import load_csv_export


@dataclass
class AlignedSession:
    directory: Path
    report: dict
    mapping: list[dict]
    times: np.ndarray
    bones: dict[str, np.ndarray]
    marker_names: list[str]
    markers: np.ndarray


def load_session(directory: Path) -> AlignedSession:
    directory = Path(directory)
    if (directory / "INCOMPLETE").exists():
        raise ValueError("This aligned session is incomplete; rerun preprocessing")
    report = json.loads((directory / "alignment.json").read_text())
    with (directory / "frame_alignment.csv").open(newline="") as stream:
        mapping = list(csv.DictReader(stream))
    timestamps = np.array([int(row["timestamp_ns"]) for row in mapping], dtype=np.int64)
    if (
        len(mapping) != report["frame_count"]
        or len(mapping) < 2
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Invalid aligned frame count or timestamps")
    if [int(row["aligned_frame"]) for row in mapping] != list(range(len(mapping))):
        raise ValueError("Aligned frame indices must be contiguous and zero-based")
    motion = load_csv_export(directory / "mocap_aligned.csv")
    bones = load_bones(directory / "mocap_aligned.csv")
    if len(motion.times) != len(mapping) or not bones:
        raise ValueError(
            "Aligned CSV and mapping have different lengths or no solved bones"
        )
    if report["source_length_units"].lower() not in {"meters", "metres", "m"}:
        raise ValueError("Paired viewer currently requires Motive positions in metres")
    names, markers = motion.point_groups["Bone Marker"]
    return AlignedSession(
        directory,
        report,
        mapping,
        (timestamps - timestamps[0]) / 1e9,
        bones,
        [name.split(":", 1)[-1] for name in names],
        markers,
    )
