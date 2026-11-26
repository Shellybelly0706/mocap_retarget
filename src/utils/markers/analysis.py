"""Reusable markers analysis helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from utils.markers.data import (
    MocapData,
)


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
        [
            robust_jump_threshold(adjacent[:, marker], jump_floor_m)
            for marker in range(marker_count)
        ]
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
        max_step = (
            float(np.nanmax(marker_steps))
            if np.any(np.isfinite(marker_steps))
            else math.nan
        )
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
