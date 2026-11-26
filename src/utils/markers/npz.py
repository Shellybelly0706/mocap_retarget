"""Reusable markers npz helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.markers.conversion import (
    ConversionResult,
)


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


def verify_npz(
    output_path: Path, expected: ConversionResult
) -> dict[str, np.ndarray | float]:
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
        raise ValueError(
            f"Unexpected valid_mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}"
        )
    if not np.array_equal(valid_mask, np.isfinite(markers).all(axis=2)):
        raise ValueError("valid_mask does not match finite marker triplets")
    if not np.isnan(markers[~valid_mask]).all():
        raise ValueError(
            "At least one invalid marker triplet contains a non-NaN coordinate"
        )
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
