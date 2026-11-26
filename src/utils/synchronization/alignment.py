"""Reusable synchronization alignment helpers."""

from __future__ import annotations

import bisect
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from utils.synchronization.clocks import (
    parse_pair,
    parse_timestamp_ns,
)
from utils.synchronization.constants import (
    NS_PER_SECOND,
)
from utils.synchronization.models import (
    AffineMap,
    AutomaticAlignment,
    EventAnchor,
    MotiveExport,
    TimedSignal,
)
from utils.synchronization.signals import (
    correlation_peak,
    normalized_valid_correlation,
    resample_signal,
)


def _nearest_frame(motive: MotiveExport, elapsed_s: float) -> tuple[int, float]:
    wanted = motive.first_frame + elapsed_s * motive.frame_rate_hz
    position = bisect.bisect_left(motive.frame_ids, wanted)
    candidates = []
    if position < len(motive.frame_ids):
        candidates.append(motive.frame_ids[position])
    if position:
        candidates.append(motive.frame_ids[position - 1])
    frame = min(candidates, key=lambda value: abs(value - wanted))
    return frame, (frame - motive.first_frame) / motive.frame_rate_hz


def automatic_event_anchors(
    motive: MotiveExport,
    motive_signal: TimedSignal,
    neon_signal: TimedSignal,
    neon_origin_ns: int,
    companion_to_ubuntu: Callable[[int], int],
    *,
    rate_hz: float,
    smoothing_seconds: float,
    window_seconds: float,
    local_search_seconds: float,
    min_correlation: float,
    min_peak_margin: float,
) -> AutomaticAlignment:
    motive_uniform = resample_signal(motive_signal, rate_hz, smoothing_seconds)
    neon_uniform = resample_signal(neon_signal, rate_hz, smoothing_seconds)
    if len(neon_uniform.values) < len(motive_uniform.values):
        raise ValueError(
            "Neon IMU 有效时长短于 Motive；" "自动模式要求 Neon 覆盖完整 Motive 录制"
        )

    global_correlation = normalized_valid_correlation(
        neon_uniform.values, motive_uniform.values
    )
    global_position, global_score, global_margin = correlation_peak(
        global_correlation, max(1, round(rate_hz * 0.75))
    )
    if global_score < min_correlation:
        raise ValueError(
            f"自动同步相关系数过低：{global_score:.3f} < {min_correlation:.3f}；"
            "请确认头部在 Motive 中被正确追踪，"
            "并录制了明显的同步头部动作"
        )
    if global_margin < min_peak_margin:
        raise ValueError(
            f"自动同步峰值不唯一：最佳峰优势 {global_margin:.3f} < "
            f"{min_peak_margin:.3f}；请使用更独特的同步动作或手工锚点"
        )

    motive_count = len(motive_uniform.values)
    window_count = min(motive_count, max(round(window_seconds * rate_hz), 100))
    step = window_count
    starts = list(range(0, max(1, motive_count - window_count + 1), step))
    last_start = motive_count - window_count
    if last_start not in starts:
        starts.append(last_start)
    starts = sorted(set(starts))
    search_samples = max(1, round(local_search_seconds * rate_hz))
    local_diagnostics: list[dict[str, object]] = []
    events: list[EventAnchor] = []
    global_std = float(np.std(motive_uniform.values))

    for start in starts:
        pattern = motive_uniform.values[start : start + window_count]
        if float(np.std(pattern)) < max(global_std * 0.12, 1e-8):
            local_diagnostics.append(
                {
                    "motive_window_start_s": float(motive_uniform.time_s[start]),
                    "accepted": False,
                    "reason": "low_motion",
                }
            )
            continue
        predicted = global_position + start
        search_left = max(0, predicted - search_samples)
        search_right = min(
            len(neon_uniform.values) - window_count,
            predicted + search_samples,
        )
        search_signal = neon_uniform.values[search_left : search_right + window_count]
        local_correlation = normalized_valid_correlation(search_signal, pattern)
        local_position, local_score, local_margin = correlation_peak(
            local_correlation, max(1, round(rate_hz * 0.15))
        )
        neon_position = search_left + local_position
        accepted = (
            local_score >= min_correlation and local_margin >= min_peak_margin / 2
        )
        diagnostic: dict[str, object] = {
            "motive_window_start_s": float(motive_uniform.time_s[start]),
            "motive_window_duration_s": window_count / rate_hz,
            "correlation": local_score,
            "peak_margin": local_margin,
            "accepted": accepted,
        }
        local_diagnostics.append(diagnostic)
        if not accepted:
            continue

        center_time = float(
            motive_uniform.time_s[start] + (window_count - 1) / (2.0 * rate_hz)
        )
        motive_frame, motive_elapsed = _nearest_frame(motive, center_time)
        neon_relative = float(
            neon_uniform.time_s[neon_position]
            + motive_elapsed
            - motive_uniform.time_s[start]
        )
        companion_ns = neon_origin_ns + round(neon_relative * NS_PER_SECOND)
        events.append(
            EventAnchor(
                motive_frame=motive_frame,
                motive_elapsed_s=motive_elapsed,
                companion_ns=companion_ns,
                ubuntu_ns=companion_to_ubuntu(companion_ns),
                source=(f"auto_imu:corr={local_score:.4f},margin={local_margin:.4f}"),
            )
        )

    # A short recording may contain only one usable window.  The global match
    # still estimates offset, but correctly leaves drift at the one-anchor
    # assumption instead of inventing two dependent anchors.
    if not events:
        center_time = float(np.mean(motive_uniform.time_s))
        motive_frame, motive_elapsed = _nearest_frame(motive, center_time)
        global_offset_s = float(
            neon_uniform.time_s[global_position] - motive_uniform.time_s[0]
        )
        companion_ns = neon_origin_ns + round(
            (global_offset_s + motive_elapsed) * NS_PER_SECOND
        )
        events.append(
            EventAnchor(
                motive_frame=motive_frame,
                motive_elapsed_s=motive_elapsed,
                companion_ns=companion_ns,
                ubuntu_ns=companion_to_ubuntu(companion_ns),
                source=f"auto_imu_global:corr={global_score:.4f}",
            )
        )

    events.sort(key=lambda item: item.motive_elapsed_s)
    # Remove a clearly inconsistent local match when at least three other
    # windows support a coherent affine mapping.
    filtered = list(events)
    while len(filtered) >= 4:
        trial = fit_affine_map(filtered)
        absolute = np.abs(np.asarray(trial.residuals_ms))
        median = float(np.median(absolute))
        threshold = max(25.0, 4.0 * median)
        worst = int(np.argmax(absolute))
        if absolute[worst] <= threshold:
            break
        del filtered[worst]

    diagnostics: dict[str, object] = {
        "method": "angular_speed_magnitude_normalized_cross_correlation",
        "resample_rate_hz": rate_hz,
        "smoothing_seconds": smoothing_seconds,
        "global_correlation": global_score,
        "global_peak_margin": global_margin,
        "global_motive_start_in_neon_seconds": float(
            neon_uniform.time_s[global_position] - motive_uniform.time_s[0]
        ),
        "local_windows": local_diagnostics,
        "accepted_anchor_count_before_outlier_filter": len(events),
        "accepted_anchor_count": len(filtered),
    }
    return AutomaticAlignment(tuple(filtered), diagnostics)


def build_event_anchors(
    motive: MotiveExport,
    direct_specs: Iterable[str],
    index_specs: Iterable[str],
    neon_times: Sequence[int] | None,
    companion_to_ubuntu: Callable[[int], int],
) -> list[EventAnchor]:
    frame_set = set(motive.frame_ids)
    events: list[EventAnchor] = []

    def add(frame: int, companion_ns: int, source: str) -> None:
        if frame not in frame_set:
            raise ValueError(
                f"同步锚点的 Motive frame {frame} 不在 CSV 中；范围为 "
                f"{motive.frame_ids[0]}..{motive.frame_ids[-1]}"
            )
        ubuntu_ns = companion_to_ubuntu(companion_ns)
        events.append(
            EventAnchor(
                motive_frame=frame,
                motive_elapsed_s=(frame - motive.first_frame) / motive.frame_rate_hz,
                companion_ns=companion_ns,
                ubuntu_ns=ubuntu_ns,
                source=source,
            )
        )

    for spec in direct_specs:
        frame, companion_ns = parse_pair(spec, parse_timestamp_ns)
        add(frame, companion_ns, f"timestamp:{companion_ns}")

    for spec in index_specs:
        if neon_times is None:
            raise ValueError("使用 --anchor-index 时必须同时提供 --neon-time-file")
        frame, index = parse_pair(spec, int)
        if index < 0 or index >= len(neon_times):
            raise ValueError(
                f"Neon 索引 {index} 越界；.time 文件有 {len(neon_times)} 个样本"
            )
        add(frame, neon_times[index], f"time_file_index:{index}")

    if not events:
        raise ValueError("至少需要一个 --anchor 或 --anchor-index 同步锚点")
    events.sort(key=lambda item: item.motive_elapsed_s)
    if len({item.motive_frame for item in events}) != len(events):
        raise ValueError("同步锚点含重复的 Motive frame")
    if len({item.companion_ns for item in events}) != len(events):
        raise ValueError("同步锚点含重复的 Neon 时间戳")
    if any(
        right.companion_ns <= left.companion_ns
        for left, right in zip(events, events[1:])
    ):
        raise ValueError("按 Motive 时间排序后，Neon 锚点时间必须严格递增")
    return events


def fit_affine_map(events: Sequence[EventAnchor]) -> AffineMap:
    if len(events) == 1:
        event = events[0]
        intercept = event.ubuntu_ns - round(event.motive_elapsed_s * NS_PER_SECOND)
        return AffineMap(slope=1.0, intercept_ns=intercept, residuals_ms=(0.0,))

    mean_x = sum(event.motive_elapsed_s for event in events) / len(events)
    reference_ns = events[0].ubuntu_ns
    relative_y = [(event.ubuntu_ns - reference_ns) / NS_PER_SECOND for event in events]
    mean_y = sum(relative_y) / len(relative_y)
    denominator = sum((event.motive_elapsed_s - mean_x) ** 2 for event in events)
    if denominator == 0:
        raise ValueError("同步锚点没有 Motive 时间跨度，无法拟合")
    slope = (
        sum(
            (event.motive_elapsed_s - mean_x) * (value - mean_y)
            for event, value in zip(events, relative_y)
        )
        / denominator
    )
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(f"拟合得到无效的时钟比例：{slope}")

    # Keep the 19-digit Unix timestamp out of floating-point arithmetic.
    intercept = reference_ns + round((mean_y - slope * mean_x) * NS_PER_SECOND)
    model = AffineMap(slope=slope, intercept_ns=intercept, residuals_ms=())
    residuals = tuple(
        (event.ubuntu_ns - model.ubuntu_ns(event.motive_elapsed_s)) / 1_000_000
        for event in events
    )
    return AffineMap(slope=slope, intercept_ns=intercept, residuals_ms=residuals)
