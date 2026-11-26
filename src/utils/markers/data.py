"""Reusable markers data helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = (
    "markers",
    "labels",
    "frame_rate",
    "frame_ids",
    "timestamps",
    "valid_mask",
)


@dataclass(frozen=True)
class MocapData:
    markers: np.ndarray
    labels: np.ndarray
    frame_rate: float
    frame_ids: np.ndarray
    timestamps: np.ndarray
    valid_mask: np.ndarray
    units: str


def scalar_text(value: np.ndarray | str | bytes) -> str:
    """Convert an NPZ scalar-like value into a plain string."""
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def load_and_validate_npz(path: Path) -> MocapData:
    """Load the six required arrays and fail loudly on inconsistent data."""
    if not path.is_file():
        raise FileNotFoundError(f"NPZ does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing_fields = [name for name in REQUIRED_FIELDS if name not in archive]
        if missing_fields:
            raise ValueError(f"NPZ is missing required fields: {missing_fields}")
        markers = np.asarray(archive["markers"], dtype=np.float64)
        labels = np.asarray(archive["labels"]).astype(str)
        frame_rate_array = np.asarray(archive["frame_rate"], dtype=np.float64)
        frame_ids = np.asarray(archive["frame_ids"])
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        valid_mask = np.asarray(archive["valid_mask"])
        units = scalar_text(archive["units"]) if "units" in archive else "m"

    if markers.ndim != 3 or markers.shape[2] != 3:
        raise ValueError(
            f"markers must have shape [frames, markers, 3], got {markers.shape}"
        )
    frame_count, marker_count, _ = markers.shape
    if labels.shape != (marker_count,):
        raise ValueError(
            f"labels shape {labels.shape} does not match {marker_count} markers"
        )
    if len(set(labels.tolist())) != marker_count:
        duplicates = sorted({x for x in labels if np.count_nonzero(labels == x) > 1})
        raise ValueError(f"Marker labels are not unique: {duplicates}")
    if any(not label.strip() for label in labels):
        raise ValueError("Marker labels must not be empty")
    if frame_rate_array.size != 1:
        raise ValueError(
            f"frame_rate must be scalar, got shape {frame_rate_array.shape}"
        )
    frame_rate = float(frame_rate_array.reshape(-1)[0])
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive and finite, got {frame_rate}")
    if frame_ids.shape != (frame_count,):
        raise ValueError(
            f"frame_ids shape {frame_ids.shape} does not match {frame_count} frames"
        )
    if not np.issubdtype(frame_ids.dtype, np.integer):
        if not np.all(np.isfinite(frame_ids)) or not np.all(
            frame_ids == np.rint(frame_ids)
        ):
            raise ValueError("frame_ids must contain integer-valued frame numbers")
        frame_ids = frame_ids.astype(np.int64)
    else:
        frame_ids = frame_ids.astype(np.int64, copy=False)
    if timestamps.shape != (frame_count,):
        raise ValueError(
            f"timestamps shape {timestamps.shape} does not match {frame_count} frames"
        )
    if valid_mask.shape != (frame_count, marker_count):
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match {(frame_count, marker_count)}"
        )
    if valid_mask.dtype != np.bool_:
        unique_mask = np.unique(valid_mask)
        if not np.all(np.isin(unique_mask, [0, 1])):
            raise ValueError("valid_mask is neither boolean nor a binary 0/1 array")
        valid_mask = valid_mask.astype(bool)

    finite_xyz = np.isfinite(markers).all(axis=2)
    any_finite_xyz = np.isfinite(markers).any(axis=2)
    if np.any(any_finite_xyz & ~finite_xyz):
        count = int(np.count_nonzero(any_finite_xyz & ~finite_xyz))
        raise ValueError(f"Found {count} partially finite XYZ samples")
    mismatch = finite_xyz != valid_mask
    if np.any(mismatch):
        raise ValueError(
            "valid_mask disagrees with finite XYZ samples at "
            f"{int(np.count_nonzero(mismatch))} frame/marker entries"
        )
    if np.any(np.isfinite(markers[~valid_mask])):
        raise ValueError(
            "Invalid marker entries must be represented by non-finite XYZ values"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contains non-finite values")
    if frame_count > 1:
        if np.any(np.diff(frame_ids) <= 0):
            raise ValueError("frame_ids must be strictly increasing")
        time_deltas = np.diff(timestamps)
        if np.any(time_deltas <= 0):
            raise ValueError("timestamps must be strictly increasing")
        # Motive exports commonly round timestamps to milliseconds. Checking
        # every delta (or only its median) would reject the valid 0.008/0.009 s
        # alternation produced by a 120 Hz stream. Compare the full timeline
        # with frame_id/frame_rate instead and allow the rounding quantum.
        expected_relative_time = (frame_ids - frame_ids[0]) / frame_rate
        observed_relative_time = timestamps - timestamps[0]
        timeline_error = np.abs(observed_relative_time - expected_relative_time)
        tolerance = max(1e-3, (1.0 / frame_rate) * 0.02)
        if float(np.max(timeline_error)) > tolerance:
            raise ValueError(
                f"timestamps are inconsistent with frame_ids/frame_rate: maximum "
                f"timeline error {float(np.max(timeline_error)):.9f}s exceeds "
                f"{tolerance:.9f}s"
            )

    normalized_units = units.strip().lower()
    if normalized_units not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError(
            f"This analysis uses metre-based thresholds, but NPZ units are {units!r}"
        )

    return MocapData(
        markers=markers,
        labels=labels,
        frame_rate=frame_rate,
        frame_ids=frame_ids,
        timestamps=timestamps,
        valid_mask=valid_mask,
        units="m",
    )
