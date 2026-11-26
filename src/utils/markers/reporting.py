"""Reusable markers reporting helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from utils.markers.analysis import (
    Analysis,
    MarkerSummary,
)
from utils.markers.data import (
    MocapData,
)
from utils.markers.selection import (
    Candidate,
)


def write_marker_summary(path: Path, summaries: Sequence[MarkerSummary]) -> None:
    fields = (
        "marker_index",
        "label",
        "valid_frames",
        "missing_rate",
        "longest_missing_frames",
        "longest_missing_seconds",
        "max_adjacent_displacement_m",
        "suspicious_jump_count",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "marker_index": item.marker_index,
                    "label": item.label,
                    "valid_frames": item.valid_frames,
                    "missing_rate": f"{item.missing_rate:.8f}",
                    "longest_missing_frames": item.longest_missing_frames,
                    "longest_missing_seconds": f"{item.longest_missing_seconds:.6f}",
                    "max_adjacent_displacement_m": (
                        f"{item.max_adjacent_displacement_m:.9f}"
                        if math.isfinite(item.max_adjacent_displacement_m)
                        else ""
                    ),
                    "suspicious_jump_count": item.suspicious_jump_count,
                }
            )


def write_candidates(path: Path, candidates: Sequence[Candidate]) -> None:
    fields = (
        "rank",
        "frame_index",
        "frame_id",
        "timestamp_seconds",
        "valid_marker_count",
        "missing_markers",
        "anomaly_status",
        "selection_reason",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "rank": item.rank,
                    "frame_index": item.frame_index,
                    "frame_id": item.frame_id,
                    "timestamp_seconds": f"{item.timestamp:.6f}",
                    "valid_marker_count": item.valid_count,
                    "missing_markers": ";".join(item.missing_markers),
                    "anomaly_status": item.anomaly_status,
                    "selection_reason": item.reason,
                }
            )


def compact_intervals(indices: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted(set(int(x) for x in indices))
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def print_report(
    data: MocapData, analysis: Analysis, candidates: Sequence[Candidate]
) -> None:
    frame_count, marker_count, _ = data.markers.shape
    mask_match = np.array_equal(data.valid_mask, np.isfinite(data.markers).all(axis=2))
    missing_xyz = int(np.count_nonzero(~np.isfinite(data.markers)))
    print("\n=== NPZ validation ===")
    print(f"markers: {data.markers.shape} {data.markers.dtype}")
    print(
        f"labels: {data.labels.shape}; unique={len(set(data.labels.tolist())) == marker_count}"
    )
    print(f"frame_rate: {data.frame_rate:g} Hz; units: {data.units}")
    print(
        f"frame_ids: {data.frame_ids.shape}; {data.frame_ids[0]} .. {data.frame_ids[-1]}"
    )
    print(
        f"timestamps: {data.timestamps.shape}; {data.timestamps[0]:.6f} .. "
        f"{data.timestamps[-1]:.6f} s"
    )
    print(f"valid_mask: {data.valid_mask.shape}; agrees with finite XYZ={mask_match}")
    print(f"non-finite coordinate values: {missing_xyz} (all excluded by valid_mask)")

    print(f"\n=== Complete marker label list ({marker_count}) ===")
    for index, label in enumerate(data.labels, 1):
        print(f"{index:02d}. {label}")

    print("\n=== Per-marker summary ===")
    print("idx  valid  missing  longest gap       max step    jumps  label")
    for item in analysis.summaries:
        max_step = (
            f"{item.max_adjacent_displacement_m * 1000:8.2f} mm"
            if math.isfinite(item.max_adjacent_displacement_m)
            else "       n/a"
        )
        print(
            f"{item.marker_index + 1:02d}  {item.valid_frames:5d}  "
            f"{item.missing_rate:7.2%}  {item.longest_missing_frames:5d} frames/"
            f"{item.longest_missing_seconds:6.3f}s  {max_step}  "
            f"{item.suspicious_jump_count:5d}  {item.label}"
        )

    jump_entries = np.argwhere(analysis.jump_mask)
    print("\n=== Large adjacent-frame jumps ===")
    if jump_entries.size == 0:
        print("None at the per-marker robust thresholds.")
    else:
        for transition, marker in jump_entries:
            frame = int(transition + 1)
            print(
                f"{data.labels[marker]}: frame {int(data.frame_ids[transition])} -> "
                f"{int(data.frame_ids[frame])}, t={data.timestamps[frame]:.6f}s, "
                f"displacement={analysis.adjacent_displacements[transition, marker]:.6f}m, "
                f"threshold={analysis.jump_thresholds[marker]:.6f}m"
            )

    print("\n=== Long consecutive missing intervals ===")
    if not analysis.long_missing_runs:
        print("None at the configured duration threshold.")
    else:
        for marker, start, end in analysis.long_missing_runs:
            final = end - 1
            print(
                f"{data.labels[marker]}: frames {int(data.frame_ids[start])}.."
                f"{int(data.frame_ids[final])}, t={data.timestamps[start]:.3f}.."
                f"{data.timestamps[final]:.3f}s, {end - start} frames/"
                f"{(end - start) / data.frame_rate:.3f}s"
            )

    print("\n=== Suspected marker-label swaps (review only; no relabeling) ===")
    if not analysis.swap_suspicions:
        print(
            "No reciprocal jump pair met the conservative swap test. "
            "This does not prove that every label is correct."
        )
    else:
        for item in analysis.swap_suspicions:
            frame = item.transition_index + 1
            print(
                f"Around frame {int(data.frame_ids[frame])} ({data.timestamps[frame]:.6f}s): "
                f"{item.first_label} <-> {item.second_label}; each new point is closer "
                f"to the other's previous point ({item.first_to_second_previous_m:.4f}m, "
                f"{item.second_to_first_previous_m:.4f}m) than to its own. Manual review required."
            )

    print("\n=== Abnormally low valid-marker counts ===")
    low_indices = np.flatnonzero(analysis.low_count_mask)
    if low_indices.size == 0:
        print(
            f"None below the lower-tail threshold of {analysis.low_count_threshold} markers."
        )
    else:
        print(
            f"Threshold: fewer than {analysis.low_count_threshold}/{marker_count} valid markers "
            f"(strictly below the empirical 5th percentile)."
        )
        for start, end in compact_intervals(low_indices):
            local_min = int(np.min(analysis.valid_counts[start : end + 1]))
            print(
                f"frames {int(data.frame_ids[start])}..{int(data.frame_ids[end])}, "
                f"t={data.timestamps[start]:.3f}..{data.timestamps[end]:.3f}s, "
                f"minimum valid={local_min}"
            )

    print(f"\n=== MoSh++ Stage-I candidates ({len(candidates)}) ===")
    for item in candidates:
        missing_text = (
            ", ".join(item.missing_markers) if item.missing_markers else "none"
        )
        print(
            f"{item.rank:02d}. frame={item.frame_id} (array index {item.frame_index}), "
            f"t={item.timestamp:.6f}s, valid={item.valid_count}/{marker_count}\n"
            f"    missing: {missing_text}\n"
            f"    anomaly: {item.anomaly_status}\n"
            f"    reason: {item.reason}"
        )


def validate_outputs(
    marker_csv: Path,
    candidate_csv: Path,
    video_path: Path,
    expected_markers: int,
    expected_candidates: int,
) -> dict[str, int | float]:
    """Re-open all three deliverables, including decoding the first MP4 frame."""
    with marker_csv.open(newline="", encoding="utf-8") as stream:
        marker_rows = list(csv.DictReader(stream))
    if len(marker_rows) != expected_markers:
        raise RuntimeError(
            f"marker_summary.csv has {len(marker_rows)} rows, expected {expected_markers}"
        )
    if len({row["label"] for row in marker_rows}) != expected_markers:
        raise RuntimeError("marker_summary.csv labels are not unique")

    with candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    if len(candidate_rows) != expected_candidates:
        raise RuntimeError(
            f"stage1_candidates.csv has {len(candidate_rows)} rows, expected {expected_candidates}"
        )

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot reopen MP4: {video_path}")
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable, first_frame = capture.read()
        if not readable or first_frame is None:
            raise RuntimeError("MP4 contains no decodable first frame")
    finally:
        capture.release()
    return {
        "marker_rows": len(marker_rows),
        "candidate_rows": len(candidate_rows),
        "video_frames": video_frames,
        "video_fps": video_fps,
        "video_width": width,
        "video_height": height,
    }
