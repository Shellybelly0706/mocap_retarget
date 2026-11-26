"""Reusable markers conversion helpers."""

from __future__ import annotations

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


def find_row(
    rows: Sequence[Sequence[str]], label: str, before: int | None = None
) -> int:
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
            "Expected exactly one Frame/Time axis row, found " f"{len(axis_candidates)}"
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
    for csv_row_number, row in enumerate(
        rows[header.axis_row + 1 :], header.axis_row + 2
    ):
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
