"""Reusable avatars marker motion helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

PROJECT_ROOT = _REPOSITORY_ROOT


DEFAULT_CSV = PROJECT_ROOT / "data" / "1029" / "Chi 2026-10-29 02.51.19 PM.csv"


# HeadTop constraint in the fitted ``test`` skeleton from Nodes.dat, converted
# into the local coordinate convention used by the exported Head quaternion.
# Nodes.dat offset: (1.19209e-7, 0.226815, -0.0163786)
# Exported local basis: (-X, Z, Y)
HEADTOP_LOCAL = np.array([-1.19209e-7, -0.0163786, 0.226815])


def parse_groups(header_rows: list[list[str]]) -> list[dict[str, object]]:
    """Return consecutive OptiTrack column groups from the seven-row header."""
    types, names, ids, quantities, axes = header_rows[2:7]
    groups: list[dict[str, object]] = []
    column = 2  # Frame and Time occupy columns 0 and 1.
    while column < len(types):
        key = (types[column], names[column], ids[column])
        end = column + 1
        while end < len(types) and (types[end], names[end], ids[end]) == key:
            end += 1
        groups.append(
            {
                "start": column,
                "end": end,
                "type": types[column],
                "name": names[column],
                "id": ids[column],
                "quantities": quantities[column:end],
                "axes": axes[column:end],
            }
        )
        column = end
    return groups


def interpolate_marker_gaps(positions: np.ndarray) -> tuple[np.ndarray, int]:
    """Linearly fill finite marker gaps, including nearest endpoint fill."""
    result = positions.copy()
    missing_before = int(np.isnan(result).sum())
    for marker in range(result.shape[1]):
        for axis in range(3):
            values = pd.Series(result[:, marker, axis])
            if values.notna().any():
                result[:, marker, axis] = values.interpolate(
                    method="linear", limit_direction="both"
                ).to_numpy()
    return result, missing_before


def load_marker_motion(csv_path: Path) -> dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(7)]

    metadata = dict(zip(header[0][0::2], header[0][1::2]))
    groups = parse_groups(header)
    table = pd.read_csv(csv_path, skiprows=7, header=None, na_values=[""])

    marker_groups = [group for group in groups if group["type"] == "Bone Marker"]
    if len(marker_groups) != 25:
        raise ValueError(
            f"Expected 25 Bone Marker groups, found {len(marker_groups)} in {csv_path}"
        )

    names = [str(group["name"]).split(":", 1)[-1] for group in marker_groups]
    positions = np.stack(
        [
            table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
            for group in marker_groups
        ],
        axis=1,
    )

    reconstructed: list[str] = []
    headtop_index = names.index("HeadTop")
    if np.isnan(positions[:, headtop_index]).all():
        head_group = next(
            group
            for group in groups
            if group["type"] == "Bone" and str(group["name"]).endswith(":Head")
        )
        head = table.iloc[
            :, int(head_group["start"]) : int(head_group["end"])
        ].to_numpy(float)
        head_rotation = Rotation.from_quat(head[:, :4])  # CSV order is XYZW.
        positions[:, headtop_index] = (
            head_rotation.apply(np.broadcast_to(HEADTOP_LOCAL, (len(table), 3)))
            + head[:, 4:7]
        )
        reconstructed.append("HeadTop")

    positions, missing_values = interpolate_marker_gaps(positions)
    if not np.isfinite(positions).all():
        bad = [
            names[i]
            for i in range(len(names))
            if not np.isfinite(positions[:, i]).all()
        ]
        raise ValueError(
            f"Markers remain entirely unavailable after reconstruction: {bad}"
        )

    times = table.iloc[:, 1].to_numpy(float)
    frame_rate = float(metadata.get("Export Frame Rate", 1.0 / np.diff(times).mean()))
    return {
        "names": names,
        "positions": positions,
        "times": times,
        "frame_rate": frame_rate,
        "metadata": metadata,
        "reconstructed": reconstructed,
        "missing_values_filled": missing_values,
    }


def marker_color(name: str, reconstructed: set[str]) -> str:
    if name in reconstructed:
        return "1 0 1 1"  # Magenta: inferred, not recorded.
    if name.startswith("L"):
        return "0.15 0.45 1 1"
    if name.startswith("R"):
        return "1 0.25 0.2 1"
    if name.startswith("Head"):
        return "1 0.8 0.1 1"
    return "0.2 0.9 0.35 1"
