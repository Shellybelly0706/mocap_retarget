#!/usr/bin/env python3
"""Inspect, visualize, and select MoSh++ Stage-I frames from marker NPZ data.

The tool is deliberately read-only with respect to the input NPZ. It does not
interpolate markers, relabel trajectories, infer anatomical correspondences,
or create marker-layout/SMPL-X vertex assignments.

Expected NPZ fields:
    markers      (frames, markers, 3), in metres
    labels       (markers,)
    frame_rate   scalar, in Hz
    frame_ids    (frames,)
    timestamps   (frames,), in seconds
    valid_mask   (frames, markers)

The output directory receives exactly three result files created by this tool:
``marker_summary.csv``, ``stage1_candidates.csv``, and
``markers_preview.mp4``. Detailed anomaly intervals are reported on stdout.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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


@dataclass(frozen=True)
class MarkerSummary:
    marker_index: int
    label: str
    valid_frames: int
    missing_rate: float
    longest_missing_frames: int
    longest_missing_seconds: float
    max_adjacent_displacement_m: float
    suspicious_jump_count: int
    jump_threshold_m: float


@dataclass(frozen=True)
class SwapSuspicion:
    transition_index: int
    first_label: str
    second_label: str
    first_to_second_previous_m: float
    second_to_first_previous_m: float


@dataclass(frozen=True)
class Candidate:
    rank: int
    frame_index: int
    frame_id: int
    timestamp: float
    valid_count: int
    missing_markers: tuple[str, ...]
    anomaly_status: str
    reason: str


@dataclass(frozen=True)
class Analysis:
    summaries: tuple[MarkerSummary, ...]
    adjacent_displacements: np.ndarray
    jump_mask: np.ndarray
    jump_thresholds: np.ndarray
    motion_score: np.ndarray
    valid_counts: np.ndarray
    low_count_mask: np.ndarray
    low_count_threshold: int
    long_missing_runs: tuple[tuple[int, int, int], ...]
    swap_suspicions: tuple[SwapSuspicion, ...]


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
        raise ValueError(f"labels shape {labels.shape} does not match {marker_count} markers")
    if len(set(labels.tolist())) != marker_count:
        duplicates = sorted({x for x in labels if np.count_nonzero(labels == x) > 1})
        raise ValueError(f"Marker labels are not unique: {duplicates}")
    if any(not label.strip() for label in labels):
        raise ValueError("Marker labels must not be empty")
    if frame_rate_array.size != 1:
        raise ValueError(f"frame_rate must be scalar, got shape {frame_rate_array.shape}")
    frame_rate = float(frame_rate_array.reshape(-1)[0])
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive and finite, got {frame_rate}")
    if frame_ids.shape != (frame_count,):
        raise ValueError(f"frame_ids shape {frame_ids.shape} does not match {frame_count} frames")
    if not np.issubdtype(frame_ids.dtype, np.integer):
        if not np.all(np.isfinite(frame_ids)) or not np.all(frame_ids == np.rint(frame_ids)):
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
        raise ValueError("Invalid marker entries must be represented by non-finite XYZ values")
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


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) runs where a 1-D boolean mask is true."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("true_runs expects a one-dimensional mask")
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def robust_jump_threshold(values: np.ndarray, absolute_floor_m: float) -> float:
    """Use an absolute floor plus a conservative per-marker robust threshold."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return absolute_floor_m
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    return max(absolute_floor_m, median + 10.0 * robust_sigma)


def compute_motion_score(adjacent: np.ndarray) -> np.ndarray:
    """Median marker displacement around every frame, without filling gaps."""
    frame_count = adjacent.shape[0] + 1
    marker_count = adjacent.shape[1]
    around = np.full((frame_count, marker_count, 2), np.nan, dtype=np.float64)
    around[1:, :, 0] = adjacent
    around[:-1, :, 1] = adjacent
    finite_count = np.sum(np.isfinite(around), axis=2)
    summed = np.nansum(around, axis=2)
    local = np.divide(
        summed,
        finite_count,
        out=np.full((frame_count, marker_count), np.nan),
        where=finite_count > 0,
    )
    motion = np.full(frame_count, np.nan, dtype=np.float64)
    for frame in range(frame_count):
        values = local[frame, np.isfinite(local[frame])]
        if values.size:
            motion[frame] = float(np.median(values))
    return motion


def detect_swap_suspicions(
    data: MocapData,
    jump_mask: np.ndarray,
    proximity_m: float,
) -> tuple[SwapSuspicion, ...]:
    """Flag only reciprocal jump transitions that resemble a two-label swap.

    This is intentionally conservative. A report is evidence for manual review,
    not an automatic relabeling decision.
    """
    suspicions: list[SwapSuspicion] = []
    for transition in np.flatnonzero(jump_mask.any(axis=1)):
        frame = int(transition + 1)
        jumping = np.flatnonzero(jump_mask[transition])
        for offset, first in enumerate(jumping):
            for second in jumping[offset + 1 :]:
                if not (
                    data.valid_mask[frame - 1, first]
                    and data.valid_mask[frame, first]
                    and data.valid_mask[frame - 1, second]
                    and data.valid_mask[frame, second]
                ):
                    continue
                first_to_second = float(
                    np.linalg.norm(
                        data.markers[frame, first] - data.markers[frame - 1, second]
                    )
                )
                second_to_first = float(
                    np.linalg.norm(
                        data.markers[frame, second] - data.markers[frame - 1, first]
                    )
                )
                own_first = float(
                    np.linalg.norm(
                        data.markers[frame, first] - data.markers[frame - 1, first]
                    )
                )
                own_second = float(
                    np.linalg.norm(
                        data.markers[frame, second] - data.markers[frame - 1, second]
                    )
                )
                if (
                    first_to_second <= proximity_m
                    and second_to_first <= proximity_m
                    and first_to_second < 0.5 * own_first
                    and second_to_first < 0.5 * own_second
                ):
                    suspicions.append(
                        SwapSuspicion(
                            transition_index=transition,
                            first_label=str(data.labels[first]),
                            second_label=str(data.labels[second]),
                            first_to_second_previous_m=first_to_second,
                            second_to_first_previous_m=second_to_first,
                        )
                    )
    return tuple(suspicions)


def analyze(
    data: MocapData,
    jump_floor_m: float,
    long_gap_seconds: float,
    swap_proximity_m: float,
) -> Analysis:
    frame_count, marker_count, _ = data.markers.shape
    adjacent = np.linalg.norm(data.markers[1:] - data.markers[:-1], axis=2)
    adjacent[~(data.valid_mask[1:] & data.valid_mask[:-1])] = np.nan

    jump_thresholds = np.array(
        [robust_jump_threshold(adjacent[:, marker], jump_floor_m) for marker in range(marker_count)]
    )
    jump_mask = adjacent > jump_thresholds[None, :]
    minimum_long_gap = max(1, int(math.ceil(long_gap_seconds * data.frame_rate)))
    long_runs: list[tuple[int, int, int]] = []
    summaries: list[MarkerSummary] = []

    for marker, label in enumerate(data.labels):
        missing_runs = true_runs(~data.valid_mask[:, marker])
        longest = max((end - start for start, end in missing_runs), default=0)
        long_runs.extend(
            (marker, start, end)
            for start, end in missing_runs
            if end - start >= minimum_long_gap
        )
        marker_steps = adjacent[:, marker]
        max_step = float(np.nanmax(marker_steps)) if np.any(np.isfinite(marker_steps)) else math.nan
        valid_frames = int(np.count_nonzero(data.valid_mask[:, marker]))
        summaries.append(
            MarkerSummary(
                marker_index=marker,
                label=str(label),
                valid_frames=valid_frames,
                missing_rate=1.0 - valid_frames / frame_count,
                longest_missing_frames=longest,
                longest_missing_seconds=longest / data.frame_rate,
                max_adjacent_displacement_m=max_step,
                suspicious_jump_count=int(np.count_nonzero(jump_mask[:, marker])),
                jump_threshold_m=float(jump_thresholds[marker]),
            )
        )

    valid_counts = data.valid_mask.sum(axis=1).astype(np.int64)
    # A data-driven lower-tail definition avoids assuming a fixed marker count.
    low_count_threshold = int(math.floor(float(np.percentile(valid_counts, 5))))
    low_count_mask = valid_counts < low_count_threshold
    motion_score = compute_motion_score(adjacent)
    swaps = detect_swap_suspicions(data, jump_mask, swap_proximity_m)
    return Analysis(
        summaries=tuple(summaries),
        adjacent_displacements=adjacent,
        jump_mask=jump_mask,
        jump_thresholds=jump_thresholds,
        motion_score=motion_score,
        valid_counts=valid_counts,
        low_count_mask=low_count_mask,
        low_count_threshold=low_count_threshold,
        long_missing_runs=tuple(long_runs),
        swap_suspicions=swaps,
    )


def dilate_frame_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    kernel = np.ones(2 * radius + 1, dtype=np.int64)
    return np.convolve(mask.astype(np.int64), kernel, mode="same") > 0


def centered_pose_distance(data: MocapData, first: int, second: int) -> float:
    """RMS pose distance on shared markers after removing per-frame translation."""
    shared = data.valid_mask[first] & data.valid_mask[second]
    if np.count_nonzero(shared) < 3:
        return 0.0
    first_xyz = data.markers[first, shared]
    second_xyz = data.markers[second, shared]
    first_xyz = first_xyz - np.median(first_xyz, axis=0)
    second_xyz = second_xyz - np.median(second_xyz, axis=0)
    return float(np.sqrt(np.mean(np.sum((first_xyz - second_xyz) ** 2, axis=1))))


def select_stage1_candidates(
    data: MocapData,
    analysis: Analysis,
    requested_count: int,
    anomaly_margin_seconds: float,
) -> tuple[Candidate, ...]:
    """Select high-quality, stable, temporally distributed, diverse frames."""
    frame_count, marker_count, _ = data.markers.shape
    if requested_count <= 0 or frame_count == 0:
        return ()
    target = min(requested_count, frame_count)

    jump_frames = np.zeros(frame_count, dtype=bool)
    transitions = np.flatnonzero(analysis.jump_mask.any(axis=1))
    jump_frames[transitions] = True
    jump_frames[np.minimum(transitions + 1, frame_count - 1)] = True
    margin = int(math.ceil(anomaly_margin_seconds * data.frame_rate))
    near_jump = dilate_frame_mask(jump_frames, margin)

    high_visibility = max(
        int(math.ceil(0.90 * marker_count)),
        int(math.floor(float(np.percentile(analysis.valid_counts, 75)))),
    )
    finite_motion = analysis.motion_score[np.isfinite(analysis.motion_score)]
    if finite_motion.size:
        stable_limit = float(np.percentile(finite_motion, 75))
        typical_motion = max(float(np.median(finite_motion)), 1e-9)
    else:
        stable_limit = math.inf
        typical_motion = 1.0

    eligible = (
        (analysis.valid_counts >= high_visibility)
        & ~analysis.low_count_mask
        & ~near_jump
        & np.isfinite(analysis.motion_score)
        & (analysis.motion_score <= stable_limit)
    )
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return ()

    visibility_quality = analysis.valid_counts / marker_count
    stability_quality = np.exp(-analysis.motion_score / typical_motion)
    quality = 0.75 * visibility_quality + 0.25 * stability_quality
    min_separation_seconds = max(
        0.25,
        (float(data.timestamps[-1]) - float(data.timestamps[0])) / (3.0 * target),
    )

    selected: list[int] = []
    selected_bins: dict[int, int] = {}
    bin_edges = np.linspace(0, frame_count, target + 1, dtype=int)
    for bin_index in range(target):
        start, end = int(bin_edges[bin_index]), int(bin_edges[bin_index + 1])
        pool = eligible_indices[(eligible_indices >= start) & (eligible_indices < end)]
        if selected:
            pool = np.array(
                [
                    frame
                    for frame in pool
                    if all(
                        abs(float(data.timestamps[frame]) - float(data.timestamps[other]))
                        >= min_separation_seconds
                        for other in selected
                    )
                ],
                dtype=np.int64,
            )
        if pool.size == 0:
            continue
        if not selected:
            best = int(pool[np.argmax(quality[pool])])
        else:
            diversity = np.array(
                [
                    min(centered_pose_distance(data, int(frame), other) for other in selected)
                    for frame in pool
                ]
            )
            scale = float(np.percentile(diversity, 75)) if diversity.size else 0.0
            diversity_score = diversity / max(scale, 1e-9)
            combined = quality[pool] + 0.20 * np.clip(diversity_score, 0.0, 2.0)
            best = int(pool[np.argmax(combined)])
        selected.append(best)
        selected_bins[best] = bin_index

    # Fill skipped bins from the global pool while keeping temporal separation.
    while len(selected) < target:
        remaining = [
            int(frame)
            for frame in eligible_indices
            if frame not in selected
            and all(
                abs(float(data.timestamps[frame]) - float(data.timestamps[other]))
                >= min_separation_seconds
                for other in selected
            )
        ]
        if not remaining:
            break
        if selected:
            diversity = np.array(
                [min(centered_pose_distance(data, frame, other) for other in selected) for frame in remaining]
            )
            scale = float(np.percentile(diversity, 75)) if diversity.size else 0.0
            combined = quality[remaining] + 0.20 * np.clip(diversity / max(scale, 1e-9), 0.0, 2.0)
        else:
            combined = quality[remaining]
        best = remaining[int(np.argmax(combined))]
        selected.append(best)
        selected_bins[best] = min(target - 1, int(best * target / frame_count))

    candidates: list[Candidate] = []
    for rank, frame in enumerate(sorted(selected), 1):
        missing = tuple(str(x) for x in data.labels[~data.valid_mask[frame]])
        if len(selected) == 1:
            diversity_text = "first selected pose"
        else:
            other_frames = [other for other in selected if other != frame]
            diversity_m = min(centered_pose_distance(data, frame, other) for other in other_frames)
            diversity_text = f"nearest selected pose distance {diversity_m:.3f} m"
        reason = (
            f"high visibility ({analysis.valid_counts[frame]}/{marker_count}); "
            f"stable local median motion {analysis.motion_score[frame] * 1000.0:.2f} mm/frame; "
            f"temporal coverage bin {selected_bins[frame] + 1}/{target}; {diversity_text}"
        )
        candidates.append(
            Candidate(
                rank=rank,
                frame_index=frame,
                frame_id=int(data.frame_ids[frame]),
                timestamp=float(data.timestamps[frame]),
                valid_count=int(analysis.valid_counts[frame]),
                missing_markers=missing,
                anomaly_status="none (jump/low-count checks passed)",
                reason=reason,
            )
        )
    return tuple(candidates)


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


class MarkerViewer:
    """Matplotlib 3-D viewer shared by interactive playback and MP4 export."""

    def __init__(
        self,
        data: MocapData,
        show_labels: bool,
        track_subject: bool,
        width_px: int,
        height_px: int,
        dpi: int,
    ) -> None:
        import matplotlib.pyplot as plt

        self.data = data
        self.show_labels = show_labels
        self.track_subject = track_subject
        self.current_frame = 0
        self.playing = True
        self.fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.scatter = self.ax.scatter([], [], [], s=18, c="#00d9ff", depthshade=False)
        self.texts = [
            self.ax.text(0.0, 0.0, 0.0, "", fontsize=5, color="#222222")
            for _ in data.labels
        ]
        self.status = self.fig.text(0.01, 0.01, "", fontsize=8)
        self.direction_artists: list[object] = []
        self._configure_axes()
        self.update(0)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _configure_axes(self) -> None:
        finite_points = self.data.markers[self.data.valid_mask]
        mins = np.min(finite_points, axis=0)
        maxs = np.max(finite_points, axis=0)
        self.fixed_center = (mins + maxs) / 2.0
        self.fixed_half_extent = max(float(np.max(maxs - mins)) / 2.0, 0.1) * 1.05
        # In a locomotion take, global equal limits can shrink a 1.7 m body to
        # a tiny cluster because the subject travels several metres. A robust
        # fixed body-scale extent lets the default camera follow translation
        # while preserving the same scale and X/Y/Z aspect on every frame.
        with np.errstate(all="ignore"):
            frame_spans = np.nanmax(self.data.markers, axis=1) - np.nanmin(
                self.data.markers, axis=1
            )
        self.tracking_half_extent = max(
            float(np.nanpercentile(frame_spans, 99.5)) / 2.0 * 1.08,
            0.1,
        )
        if self.track_subject:
            first_valid = self.data.valid_mask[0]
            center = np.median(self.data.markers[0, first_valid], axis=0)
            half_extent = self.tracking_half_extent
        else:
            center = self.fixed_center
            half_extent = self.fixed_half_extent
        self._set_view_center(center, half_extent)
        self.ax.set_box_aspect((1.0, 1.0, 1.0))
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.grid(True, alpha=0.35)
        self.fig.subplots_adjust(left=0.02, right=0.96, bottom=0.06, top=0.92)

    def _set_view_center(self, center: np.ndarray, half_extent: float) -> None:
        self.ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
        self.ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
        self.ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
        for artist in self.direction_artists:
            artist.remove()
        self.direction_artists.clear()
        arrow_origin = center - 0.82 * half_extent
        arrow_length = 0.18 * half_extent
        for direction, color, label in (
            ((1, 0, 0), "#d62728", "+X"),
            ((0, 1, 0), "#2ca02c", "+Y"),
            ((0, 0, 1), "#1f77b4", "+Z"),
        ):
            arrow = self.ax.quiver(
                *arrow_origin,
                *(np.asarray(direction) * arrow_length),
                color=color,
                arrow_length_ratio=0.18,
            )
            endpoint = arrow_origin + np.asarray(direction) * arrow_length * 1.12
            label_artist = self.ax.text(
                *endpoint, label, color=color, fontsize=8, fontweight="bold"
            )
            self.direction_artists.extend((arrow, label_artist))

    def update(self, frame: int) -> list[object]:
        frame = int(np.clip(frame, 0, self.data.markers.shape[0] - 1))
        self.current_frame = frame
        valid = self.data.valid_mask[frame]
        xyz = self.data.markers[frame, valid]
        if self.track_subject and xyz.size:
            self._set_view_center(np.median(xyz, axis=0), self.tracking_half_extent)
        if xyz.size:
            self.scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            empty = np.array([], dtype=np.float64)
            self.scatter._offsets3d = (empty, empty, empty)
        for marker, text_artist in enumerate(self.texts):
            visible = bool(valid[marker] and self.show_labels)
            text_artist.set_visible(visible)
            if visible:
                point = self.data.markers[frame, marker]
                text_artist.set_position((point[0], point[1]))
                text_artist.set_3d_properties(point[2], zdir="z")
                text_artist.set_text(str(self.data.labels[marker]))
        self.ax.set_title(
            f"Frame {int(self.data.frame_ids[frame])}  |  "
            f"t = {float(self.data.timestamps[frame]):.3f} s  |  "
            f"valid = {int(np.count_nonzero(valid))}/{len(valid)}"
        )
        self.status.set_text(
            "Space/P: play-pause   ←/→: previous/next frame   "
            "L: toggle labels   Home/End: first/last"
        )
        return [self.scatter, self.ax.title, self.status, *self.texts]

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key in {" ", "space", "p"}:
            self.playing = not self.playing
            if hasattr(self, "animation"):
                if self.playing:
                    self.animation.event_source.start()
                else:
                    self.animation.event_source.stop()
        elif key == "right":
            self.playing = False
            if hasattr(self, "animation"):
                self.animation.event_source.stop()
            self.update(self.current_frame + 1)
            self.fig.canvas.draw_idle()
        elif key == "left":
            self.playing = False
            if hasattr(self, "animation"):
                self.animation.event_source.stop()
            self.update(self.current_frame - 1)
            self.fig.canvas.draw_idle()
        elif key and key.lower() == "l":
            self.show_labels = not self.show_labels
            self.update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif key == "home":
            self.update(0)
            self.fig.canvas.draw_idle()
        elif key == "end":
            self.update(self.data.markers.shape[0] - 1)
            self.fig.canvas.draw_idle()

    def show(self) -> None:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        def advance(_: int) -> list[object]:
            if self.playing:
                next_frame = (self.current_frame + 1) % self.data.markers.shape[0]
                return self.update(next_frame)
            return []

        interval_ms = 1000.0 / self.data.frame_rate
        self.animation = animation.FuncAnimation(
            self.fig,
            advance,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def export_mp4(
    data: MocapData,
    destination: Path,
    show_labels: bool,
    track_subject: bool,
    video_fps: float,
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[int, int, float]:
    """Render a real-time-duration MP4 via OpenCV, sampling only the preview."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("MP4 export requires the installed OpenCV (cv2) package") from error

    import matplotlib.pyplot as plt

    stride = max(1, int(round(data.frame_rate / video_fps)))
    actual_fps = data.frame_rate / stride
    frame_indices = list(range(0, data.markers.shape[0], stride))
    if frame_indices[-1] != data.markers.shape[0] - 1:
        frame_indices.append(data.markers.shape[0] - 1)

    viewer = MarkerViewer(data, show_labels, track_subject, width_px, height_px, dpi)
    viewer.fig.canvas.draw()
    canvas_width, canvas_height = viewer.fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        plt.close(viewer.fig)
        raise RuntimeError("OpenCV could not open an MP4 writer with the mp4v codec")

    print(
        f"Rendering MP4: {len(frame_indices)} preview frames, stride={stride}, "
        f"fps={actual_fps:g}, size={canvas_width}x{canvas_height}",
        flush=True,
    )
    try:
        progress_step = max(1, len(frame_indices) // 10)
        for output_index, source_frame in enumerate(frame_indices):
            viewer.update(source_frame)
            viewer.fig.canvas.draw()
            rgba = np.asarray(viewer.fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
            if (output_index + 1) % progress_step == 0 or output_index + 1 == len(frame_indices):
                print(
                    f"  MP4 progress: {output_index + 1}/{len(frame_indices)}",
                    flush=True,
                )
    finally:
        writer.release()
        plt.close(viewer.fig)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer did not create a readable file: {destination}")
    return len(frame_indices), stride, actual_fps


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


def print_report(data: MocapData, analysis: Analysis, candidates: Sequence[Candidate]) -> None:
    frame_count, marker_count, _ = data.markers.shape
    mask_match = np.array_equal(data.valid_mask, np.isfinite(data.markers).all(axis=2))
    missing_xyz = int(np.count_nonzero(~np.isfinite(data.markers)))
    print("\n=== NPZ validation ===")
    print(f"markers: {data.markers.shape} {data.markers.dtype}")
    print(f"labels: {data.labels.shape}; unique={len(set(data.labels.tolist())) == marker_count}")
    print(f"frame_rate: {data.frame_rate:g} Hz; units: {data.units}")
    print(f"frame_ids: {data.frame_ids.shape}; {data.frame_ids[0]} .. {data.frame_ids[-1]}")
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
    print(
        "idx  valid  missing  longest gap       max step    jumps  label"
    )
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
        print(f"None below the lower-tail threshold of {analysis.low_count_threshold} markers.")
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
        missing_text = ", ".join(item.missing_markers) if item.missing_markers else "none"
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and visualize a marker NPZ, report trajectory anomalies, "
            "and select MoSh++ Stage-I candidate frames."
        )
    )
    parser.add_argument("input_npz", type=Path, help="Input marker NPZ (never modified)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Result directory")
    parser.add_argument(
        "--candidate-count", type=int, default=12, help="Maximum Stage-I frames (default: 12)"
    )
    parser.add_argument(
        "--jump-threshold-m",
        type=float,
        default=0.08,
        help="Minimum suspicious adjacent-frame displacement in metres (default: 0.08)",
    )
    parser.add_argument(
        "--long-gap-seconds",
        type=float,
        default=1.0,
        help="Minimum missing-run duration reported as long (default: 1.0)",
    )
    parser.add_argument(
        "--swap-proximity-m",
        type=float,
        default=0.03,
        help="Maximum cross-label distance used by the conservative swap test (default: 0.03)",
    )
    parser.add_argument(
        "--anomaly-margin-seconds",
        type=float,
        default=0.25,
        help="Candidate exclusion margin around jump frames (default: 0.25)",
    )
    parser.add_argument(
        "--video-fps", type=float, default=30.0, help="Target preview FPS (default: 30)"
    )
    parser.add_argument("--video-width", type=int, default=960, help="MP4 width (default: 960)")
    parser.add_argument("--video-height", type=int, default=720, help="MP4 height (default: 720)")
    parser.add_argument("--dpi", type=int, default=100, help="Matplotlib render DPI (default: 100)")
    parser.add_argument(
        "--labels-off",
        action="store_true",
        help="Start without point labels and export the MP4 without them",
    )
    parser.add_argument(
        "--fixed-view",
        action="store_true",
        help=(
            "Use one global equal-axis view instead of following the subject "
            "with a fixed body-scale view"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive viewer after writing and validating outputs",
    )
    args = parser.parse_args(argv)
    if args.candidate_count < 1:
        parser.error("--candidate-count must be at least 1")
    if args.jump_threshold_m <= 0 or args.long_gap_seconds <= 0:
        parser.error("jump and long-gap thresholds must be positive")
    if args.swap_proximity_m <= 0 or args.anomaly_margin_seconds < 0:
        parser.error("swap proximity must be positive and anomaly margin non-negative")
    if args.video_fps <= 0 or args.video_width < 320 or args.video_height < 240 or args.dpi <= 0:
        parser.error("video FPS/dimensions/DPI are invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Select a non-GUI backend before pyplot is imported during ordinary batch use.
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    input_path = args.input_npz.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data = load_and_validate_npz(input_path)
    analysis = analyze(
        data,
        jump_floor_m=args.jump_threshold_m,
        long_gap_seconds=args.long_gap_seconds,
        swap_proximity_m=args.swap_proximity_m,
    )
    candidates = select_stage1_candidates(
        data,
        analysis,
        requested_count=args.candidate_count,
        anomaly_margin_seconds=args.anomaly_margin_seconds,
    )
    print_report(data, analysis, candidates)
    if len(candidates) < args.candidate_count:
        print(
            f"\nOnly {len(candidates)} candidates met the quality, anomaly, stability, "
            f"diversity, and temporal-separation constraints; no low-quality frames were forced."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_csv = output_dir / "marker_summary.csv"
    candidate_csv = output_dir / "stage1_candidates.csv"
    video_path = output_dir / "markers_preview.mp4"
    write_marker_summary(marker_csv, analysis.summaries)
    write_candidates(candidate_csv, candidates)
    preview_frames, stride, preview_fps = export_mp4(
        data,
        video_path,
        show_labels=not args.labels_off,
        track_subject=not args.fixed_view,
        video_fps=args.video_fps,
        width_px=args.video_width,
        height_px=args.video_height,
        dpi=args.dpi,
    )
    verified = validate_outputs(
        marker_csv,
        candidate_csv,
        video_path,
        expected_markers=len(data.labels),
        expected_candidates=len(candidates),
    )
    print("\n=== Output verification ===")
    print(f"marker_summary.csv: reopened, {verified['marker_rows']} marker rows")
    print(f"stage1_candidates.csv: reopened, {verified['candidate_rows']} candidate rows")
    print(
        f"markers_preview.mp4: reopened and decoded; {verified['video_frames']} frames, "
        f"{verified['video_fps']:.3f} fps, {verified['video_width']}x{verified['video_height']}"
    )
    print(
        f"Preview sampling: {preview_frames} rendered frames, source stride={stride}, "
        f"effective fps={preview_fps:g}"
    )
    print(f"Output directory: {output_dir}")

    if args.show:
        viewer = MarkerViewer(
            data,
            show_labels=not args.labels_off,
            track_subject=not args.fixed_view,
            width_px=args.video_width,
            height_px=args.video_height,
            dpi=args.dpi,
        )
        viewer.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
