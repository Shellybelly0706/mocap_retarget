#!/usr/bin/env python3
"""Convert named Motive Bone Marker trajectories to a MoSh++ NPZ file.

The Motive CSV header is discovered by its semantic row labels rather than by
fixed row or column numbers. Missing XYZ samples remain missing (all three
coordinates are stored as NaN); the script never interpolates trajectories.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
}


@dataclass(frozen=True)
class HeaderRows:
    type_row: int
    name_row: int
    quantity_row: int
    axis_row: int


@dataclass(frozen=True)
class ConversionResult:
    markers: np.ndarray
    labels: np.ndarray
    frame_rate: float
    frame_ids: np.ndarray
    timestamps: np.ndarray
    valid_mask: np.ndarray
    source_units: str
    coordinate_space: str


def normalized(value: str) -> str:
    return value.strip().lower()


def find_row(rows: Sequence[Sequence[str]], label: str, before: int | None = None) -> int:
    """Find a header row by a semantic label appearing in either first column."""
    stop = len(rows) if before is None else before
    wanted = normalized(label)
    matches = [
        index
        for index, row in enumerate(rows[:stop])
        if any(normalized(cell) == wanted for cell in row[:2])
    ]
    if not matches:
        raise ValueError(f"Could not find Motive header row {label!r}")
    return matches[-1]


def discover_header(rows: Sequence[Sequence[str]]) -> HeaderRows:
    axis_candidates = [
        index
        for index, row in enumerate(rows)
        if len(row) >= 2
        and normalized(row[0]) == "frame"
        and normalized(row[1]) in {"time", "time (seconds)"}
    ]
    if len(axis_candidates) != 1:
        raise ValueError(
            "Expected exactly one Frame/Time axis row, found "
            f"{len(axis_candidates)}"
        )
    axis_row = axis_candidates[0]
    type_row = find_row(rows, "Type", before=axis_row)
    name_row = find_row(rows, "Name", before=axis_row)

    quantity_candidates = [
        index
        for index in range(max(type_row, name_row) + 1, axis_row)
        if any(normalized(cell) == "position" for cell in rows[index])
    ]
    if not quantity_candidates:
        raise ValueError("Could not find the Rotation/Position quantity row")
    quantity_row = quantity_candidates[-1]
    if not (type_row < name_row < quantity_row < axis_row):
        raise ValueError(
            "Unexpected Motive header order: "
            f"Type={type_row + 1}, Name={name_row + 1}, "
            f"Quantity={quantity_row + 1}, Axis={axis_row + 1}"
        )
    return HeaderRows(type_row, name_row, quantity_row, axis_row)


def metadata_value(rows: Sequence[Sequence[str]], key: str, before: int) -> str:
    wanted = normalized(key)
    for row in rows[:before]:
        for index, cell in enumerate(row[:-1]):
            if normalized(cell) == wanted:
                return row[index + 1].strip()
    raise ValueError(f"CSV metadata does not contain {key!r}")


def get_frame_rate(rows: Sequence[Sequence[str]], before: int) -> float:
    for key in ("Export Frame Rate", "Capture Frame Rate"):
        try:
            rate = float(metadata_value(rows, key, before))
        except ValueError:
            continue
        if np.isfinite(rate) and rate > 0:
            return rate
    raise ValueError("CSV has no positive Export or Capture Frame Rate")


def get_unit_scale(rows: Sequence[Sequence[str]], before: int) -> tuple[str, float]:
    source_units = metadata_value(rows, "Length Units", before)
    scale = UNIT_TO_METERS.get(normalized(source_units))
    if scale is None:
        supported = ", ".join(sorted(UNIT_TO_METERS))
        raise ValueError(
            f"Unsupported Length Units {source_units!r}; supported spellings: {supported}"
        )
    return source_units, scale


def padded_cell(row: Sequence[str], column: int) -> str:
    return row[column].strip() if column < len(row) else ""


def discover_marker_columns(
    rows: Sequence[Sequence[str]], header: HeaderRows, skeleton: str
) -> OrderedDict[str, tuple[int, int, int]]:
    types = rows[header.type_row]
    names = rows[header.name_row]
    quantities = rows[header.quantity_row]
    axes = rows[header.axis_row]
    width = max(map(len, (types, names, quantities, axes)))
    prefix = f"{skeleton}:"
    marker_axes: OrderedDict[str, dict[str, int]] = OrderedDict()

    for column in range(width):
        if padded_cell(types, column) != "Bone Marker":
            continue
        name = padded_cell(names, column)
        if not name.startswith(prefix) or not name[len(prefix) :].strip():
            continue
        if padded_cell(quantities, column) != "Position":
            continue
        axis = padded_cell(axes, column).upper()
        if axis not in {"X", "Y", "Z"}:
            continue
        columns = marker_axes.setdefault(name, {})
        if axis in columns:
            raise ValueError(f"Marker {name!r} has duplicate {axis} columns")
        columns[axis] = column

    if not marker_axes:
        raise ValueError(
            f"No named Bone Marker position columns found for {skeleton!r}"
        )

    incomplete = {
        name: sorted(columns)
        for name, columns in marker_axes.items()
        if set(columns) != {"X", "Y", "Z"}
    }
    if incomplete:
        raise ValueError(f"Markers without complete XYZ columns: {incomplete}")

    return OrderedDict(
        (name, (columns["X"], columns["Y"], columns["Z"]))
        for name, columns in marker_axes.items()
    )


def parse_float(cell: str) -> float:
    value = cell.strip()
    return float(value) if value else np.nan


def convert_csv(csv_path: Path, skeleton: str) -> ConversionResult:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    header = discover_header(rows)
    marker_columns = discover_marker_columns(rows, header, skeleton)
    frame_rate = get_frame_rate(rows, header.type_row)
    source_units, meters_per_unit = get_unit_scale(rows, header.type_row)
    try:
        coordinate_space = metadata_value(rows, "Coordinate Space", header.type_row)
    except ValueError:
        coordinate_space = ""

    samples: list[np.ndarray] = []
    frame_ids: list[int] = []
    timestamps: list[float] = []
    for csv_row_number, row in enumerate(rows[header.axis_row + 1 :], header.axis_row + 2):
        if not row or not any(cell.strip() for cell in row):
            continue
        try:
            frame_id = int(padded_cell(row, 0))
            timestamp = float(padded_cell(row, 1))
        except (ValueError, IndexError) as error:
            raise ValueError(
                f"Invalid Frame/Time values on CSV row {csv_row_number}"
            ) from error

        frame_markers = np.array(
            [
                [parse_float(padded_cell(row, column)) for column in columns]
                for columns in marker_columns.values()
            ],
            dtype=np.float64,
        )
        complete = np.isfinite(frame_markers).all(axis=1)
        frame_markers[~complete] = np.nan
        samples.append(frame_markers * meters_per_unit)
        frame_ids.append(frame_id)
        timestamps.append(timestamp)

    if not samples:
        raise ValueError("CSV contains no data frames")

    markers = np.stack(samples, axis=0)
    valid_mask = np.isfinite(markers).all(axis=2)
    return ConversionResult(
        markers=markers,
        labels=np.asarray(list(marker_columns), dtype=np.str_),
        frame_rate=frame_rate,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        valid_mask=valid_mask,
        source_units=source_units,
        coordinate_space=coordinate_space,
    )


def default_output_path(csv_path: Path) -> Path:
    if normalized(csv_path.parent.name) == "csv":
        return csv_path.parent.parent / "npz" / f"{csv_path.stem}.npz"
    return csv_path.with_suffix(".npz")


def write_npz(output_path: Path, result: ConversionResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        markers=result.markers,
        labels=result.labels,
        frame_rate=np.asarray(result.frame_rate, dtype=np.float64),
        frame_ids=result.frame_ids,
        timestamps=result.timestamps,
        valid_mask=result.valid_mask,
        units=np.asarray("m"),
        source_units=np.asarray(result.source_units),
        coordinate_space=np.asarray(result.coordinate_space),
    )


def verify_npz(output_path: Path, expected: ConversionResult) -> dict[str, np.ndarray | float]:
    required = {
        "markers",
        "labels",
        "frame_rate",
        "frame_ids",
        "timestamps",
        "valid_mask",
        "units",
    }
    with np.load(output_path, allow_pickle=False) as archive:
        missing_fields = required.difference(archive.files)
        if missing_fields:
            raise ValueError(f"NPZ is missing fields: {sorted(missing_fields)}")
        markers = archive["markers"]
        labels = archive["labels"]
        frame_rate = float(archive["frame_rate"])
        frame_ids = archive["frame_ids"]
        timestamps = archive["timestamps"]
        valid_mask = archive["valid_mask"]
        units = str(archive["units"])

    frame_count, marker_count = expected.markers.shape[:2]
    if markers.shape != (frame_count, marker_count, 3):
        raise ValueError(f"Unexpected markers shape: {markers.shape}")
    if labels.shape != (marker_count,) or len(set(labels.tolist())) != marker_count:
        raise ValueError("Marker labels are not one-dimensional and unique")
    if valid_mask.shape != (frame_count, marker_count) or valid_mask.dtype != np.bool_:
        raise ValueError(f"Unexpected valid_mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}")
    if not np.array_equal(valid_mask, np.isfinite(markers).all(axis=2)):
        raise ValueError("valid_mask does not match finite marker triplets")
    if not np.isnan(markers[~valid_mask]).all():
        raise ValueError("At least one invalid marker triplet contains a non-NaN coordinate")
    if not np.array_equal(labels, expected.labels):
        raise ValueError("Marker order or labels changed during NPZ round-trip")
    if not np.array_equal(frame_ids, expected.frame_ids):
        raise ValueError("Frame IDs changed during NPZ round-trip")
    if not np.array_equal(timestamps, expected.timestamps):
        raise ValueError("Timestamps changed during NPZ round-trip")
    if not np.isclose(frame_rate, expected.frame_rate) or frame_rate <= 0:
        raise ValueError(f"Unexpected frame rate: {frame_rate}")
    if units != "m":
        raise ValueError(f"Expected output units 'm', found {units!r}")

    return {
        "valid_per_frame": valid_mask.sum(axis=1),
        "missing_rate_per_marker": 1.0 - valid_mask.mean(axis=0),
        "frame_rate": frame_rate,
    }


def print_validation(
    output_path: Path,
    result: ConversionResult,
    stats: dict[str, np.ndarray | float],
    frame_counts: str,
) -> None:
    valid_per_frame = np.asarray(stats["valid_per_frame"])
    missing_rates = np.asarray(stats["missing_rate_per_marker"])
    print(f"Wrote: {output_path.resolve()}")
    print(f"markers shape: {result.markers.shape}")
    print(f"labels: {len(result.labels)} unique")
    print(f"frame rate: {stats['frame_rate']:.6g} Hz")
    print(f"source units: {result.source_units}; output units: m")
    print(f"coordinate space: {result.coordinate_space or 'unspecified'}")
    print(f"missing values: {int((~result.valid_mask).sum())} marker-frames, stored as NaN")
    print(
        "valid markers per frame: "
        f"min={valid_per_frame.min()}, median={np.median(valid_per_frame):.1f}, "
        f"max={valid_per_frame.max()}"
    )
    if frame_counts == "all":
        print("frame_id,timestamp,valid_markers")
        for frame_id, timestamp, count in zip(
            result.frame_ids, result.timestamps, valid_per_frame
        ):
            print(f"{frame_id},{timestamp:.6f},{count}")

    print("marker,valid_frames,missing_rate")
    for marker_index, (label, missing_rate) in enumerate(zip(result.labels, missing_rates)):
        valid_frames = int(result.valid_mask[:, marker_index].sum())
        print(f"{label},{valid_frames},{missing_rate:.6%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Motive CSV to convert")
    parser.add_argument("-o", "--output", type=Path, help="Output NPZ path")
    parser.add_argument(
        "--skeleton",
        default="Skeleton 003",
        help="Exact skeleton prefix to select (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-counts",
        choices=("all", "summary"),
        default="all",
        help="Print every frame's valid count or only its summary (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(csv_path)
    )
    result = convert_csv(csv_path, args.skeleton)
    write_npz(output_path, result)
    stats = verify_npz(output_path, result)
    print_validation(output_path, result, stats, args.frame_counts)


if __name__ == "__main__":
    main()
