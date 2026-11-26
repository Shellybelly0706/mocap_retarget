"""Reusable markers selection helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from utils.markers.analysis import (
    Analysis,
)
from utils.markers.data import (
    MocapData,
)


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
                        abs(
                            float(data.timestamps[frame])
                            - float(data.timestamps[other])
                        )
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
                    min(
                        centered_pose_distance(data, int(frame), other)
                        for other in selected
                    )
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
                [
                    min(
                        centered_pose_distance(data, frame, other) for other in selected
                    )
                    for frame in remaining
                ]
            )
            scale = float(np.percentile(diversity, 75)) if diversity.size else 0.0
            combined = quality[remaining] + 0.20 * np.clip(
                diversity / max(scale, 1e-9), 0.0, 2.0
            )
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
            diversity_m = min(
                centered_pose_distance(data, frame, other) for other in other_frames
            )
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
