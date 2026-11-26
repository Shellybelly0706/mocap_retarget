"""Reusable optitrack csv helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

PROJECT_ROOT = _REPOSITORY_ROOT


DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "1105" / "csv"


@dataclass
class CsvExport:
    path: Path
    times: np.ndarray
    frame_rate: float
    point_groups: dict[str, tuple[list[str], np.ndarray]]
    rigid_names: list[str]
    rigid_positions: np.ndarray
    rigid_quaternions: np.ndarray
    bone_names: list[str]
    bone_positions: np.ndarray


def parse_csv_groups(header: list[list[str]]) -> list[dict[str, object]]:
    """Parse consecutive object groups from the seven-row Motive header."""
    types, names, ids, quantities, axes = header[2:7]
    groups: list[dict[str, object]] = []
    column = 2
    while column < len(types):
        key = (types[column].strip(), names[column].strip(), ids[column].strip())
        end = column + 1
        while end < len(types):
            candidate = (
                types[end].strip(),
                names[end].strip(),
                ids[end].strip(),
            )
            if candidate != key:
                break
            end += 1
        groups.append(
            {
                "start": column,
                "end": end,
                "type": key[0],
                "name": key[1],
                "quantities": quantities[column:end],
                "axes": axes[column:end],
            }
        )
        column = end
    return groups


def load_csv_export(path: Path) -> CsvExport:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(7)]
    metadata = dict(zip(header[0][0::2], header[0][1::2]))
    groups = parse_csv_groups(header)
    table = pd.read_csv(path, skiprows=7, header=None, na_values=[""])
    times = table.iloc[:, 1].to_numpy(float)

    points: dict[str, tuple[list[str], np.ndarray]] = {}
    for group_type in ("Bone Marker", "Rigid Body Marker", "Marker"):
        selected = [group for group in groups if group["type"] == group_type]
        names = [str(group["name"]) for group in selected]
        values = [
            table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
            for group in selected
        ]
        if any(value.shape[1] != 3 for value in values):
            raise ValueError(f"{group_type} does not contain XYZ positions")
        positions = (
            np.stack(values, axis=1)
            if values
            else np.empty((len(table), 0, 3), dtype=float)
        )
        points[group_type] = (names, positions)

    bones = [group for group in groups if group["type"] == "Bone"]
    bone_values = [
        table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
        for group in bones
    ]
    if any(value.shape[1] != 7 for value in bone_values):
        raise ValueError("CSV bones must contain quaternion XYZW and position XYZ")
    bone_positions = (
        np.stack([value[:, 4:7] for value in bone_values], axis=1)
        if bone_values
        else np.empty((len(table), 0, 3), dtype=float)
    )

    rigid = [group for group in groups if group["type"] == "Rigid Body"]
    rigid_values = [
        table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
        for group in rigid
    ]
    if any(value.shape[1] != 7 for value in rigid_values):
        raise ValueError(
            "CSV rigid bodies must contain quaternion XYZW and position XYZ"
        )
    if rigid_values:
        rigid_array = np.stack(rigid_values, axis=1)
        rigid_positions = rigid_array[:, :, 4:7]
        rigid_quaternions = rigid_array[:, :, :4][:, :, [3, 0, 1, 2]]
    else:
        rigid_positions = np.empty((len(table), 0, 3), dtype=float)
        rigid_quaternions = np.empty((len(table), 0, 4), dtype=float)

    fallback_rate = 1.0 / float(np.median(np.diff(times)))
    return CsvExport(
        path=path,
        times=times,
        frame_rate=float(metadata.get("Export Frame Rate", fallback_rate)),
        point_groups=points,
        rigid_names=[str(group["name"]) for group in rigid],
        rigid_positions=rigid_positions,
        rigid_quaternions=rigid_quaternions,
        bone_names=[str(group["name"]) for group in bones],
        bone_positions=bone_positions,
    )


def available_takes(csv_dir: Path) -> list[Path]:
    return sorted(csv_dir.glob("*.csv"))


def resolve_take(csv_dir: Path, selector: str) -> Path:
    supplied_path = Path(selector).expanduser()
    if supplied_path.is_file():
        return supplied_path.resolve()

    takes = available_takes(csv_dir)
    if not takes:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")
    if selector.isdigit() and 1 <= int(selector) <= len(takes):
        return takes[int(selector) - 1]

    exact = [path for path in takes if path.name == selector or path.stem == selector]
    matches = exact or [path for path in takes if selector.lower() in path.stem.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Take selector {selector!r} matches multiple CSV files")
    raise ValueError(f"Cannot resolve take selector {selector!r}; use --list")
