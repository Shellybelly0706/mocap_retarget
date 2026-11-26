"""Reusable optitrack c3d helpers."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

REPOSITORY_ROOT = _REPOSITORY_ROOT


DEFAULT_C3D_READER = REPOSITORY_ROOT / "src/moshpp/src/moshpp/tools/c3d.py"


UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
}


@dataclass(frozen=True)
class C3DMotion:
    path: Path
    labels: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    frame_numbers: np.ndarray
    frame_rate: float
    source_units: str


def load_c3d_module(reader_path: Path):
    if not reader_path.is_file():
        raise FileNotFoundError(f"Bundled C3D reader does not exist: {reader_path}")
    spec = importlib.util.spec_from_file_location("_local_moshpp_c3d", reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load C3D reader: {reader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_label(value: object, index: int) -> str:
    if isinstance(value, bytes):
        label = value.decode("utf-8", errors="replace")
    else:
        label = str(value)
    return label.rstrip("\x00 ").strip() or f"Point_{index + 1}"


def load_c3d(path: Path, reader_path: Path = DEFAULT_C3D_READER) -> C3DMotion:
    if not path.is_file():
        raise FileNotFoundError(path)
    c3d = load_c3d_module(reader_path)
    with path.open("rb") as stream, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="missing parameter POINT:DESCRIPTIONS"
        )
        warnings.filterwarnings("ignore", message="No analog data found in file.*")
        reader = c3d.Reader(stream)
        labels = tuple(
            clean_label(value, index) for index, value in enumerate(reader.point_labels)
        )
        units_parameter = reader.get("POINT:UNITS")
        source_units = (
            units_parameter.string_value.rstrip("\x00 ").strip()
            if units_parameter is not None
            else ""
        )
        scale = UNIT_TO_METERS.get(source_units.lower())
        if scale is None:
            raise ValueError(
                f"Unsupported or missing C3D POINT:UNITS {source_units!r}; "
                f"supported units: {sorted(UNIT_TO_METERS)}"
            )
        frame_rate = float(reader.point_rate)
        frames = list(reader.read_frames(copy=True))

    if not frames:
        raise ValueError(f"C3D has no point frames: {path}")
    positions = np.stack([points[:, :3] for _, points, _ in frames]).astype(float)
    residuals = np.stack([points[:, 3] for _, points, _ in frames])
    valid = np.isfinite(positions).all(axis=2) & (residuals >= 0.0)
    positions *= scale
    positions[~valid] = np.nan
    if positions.shape[1] != len(labels):
        raise ValueError(
            f"C3D has {positions.shape[1]} point tracks but {len(labels)} labels"
        )
    if frame_rate <= 0 or not np.isfinite(frame_rate):
        raise ValueError(f"Invalid C3D point rate: {frame_rate}")
    return C3DMotion(
        path=path,
        labels=labels,
        positions=positions,
        valid=valid,
        frame_numbers=np.asarray([number for number, _, _ in frames], dtype=int),
        frame_rate=frame_rate,
        source_units=source_units,
    )


def print_summary(motion: C3DMotion) -> None:
    frame_count, marker_count = motion.positions.shape[:2]
    print(f"C3D: {motion.path}")
    print(
        f"Frames: {frame_count} @ {motion.frame_rate:g} Hz; "
        f"duration: {(frame_count - 1) / motion.frame_rate:.3f} s"
    )
    print(
        f"Markers: {marker_count}; source units: {motion.source_units}; viewer units: m"
    )
    print(
        f"Valid samples: {int(motion.valid.sum())}/{motion.valid.size} "
        f"({100.0 * motion.valid.mean():.1f}%)"
    )
    print("Marker labels:")
    for index, label in enumerate(motion.labels, start=1):
        print(f"  {index:02d}: {label}")
