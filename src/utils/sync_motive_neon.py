#!/usr/bin/env python3
"""Map exported Motive frames onto the Ubuntu and Neon Companion clocks.

This is an offline synchronizer for the following acquisition setup:

* Pupil Labs Time Echo is measured continuously between Neon Companion and
  Ubuntu.  ``src/neon/clock_offset_monitor.py`` writes ``clock_offsets.jsonl``.
* Motive and Neon cannot communicate during capture.
* One or more physical events are identified in both recordings after export.

For every common event, provide the Motive export frame and either its Neon
Companion timestamp or its zero-based index in a Neon ``*.time`` file.  One
event estimates a constant offset.  Two or more events fit an affine clock map
and therefore also estimate clock-rate drift.

Examples
--------
Using direct Neon timestamps (integer values are Unix nanoseconds)::

    python src/utils/sync_motive_neon.py \
      --motive-csv mocap.csv \
      --offset-log src/neon/time_sync/SESSION/clock_offsets.jsonl \
      --anchor 120=1795515870123456789 \
      --anchor 1700=1795515883298765432 \
      --output-csv motive_ubuntu_timestamps.csv

Using indices from a Neon scene-camera or IMU time file::

    python src/utils/sync_motive_neon.py \
      --motive-csv mocap.csv \
      --offset-log src/neon/time_sync/SESSION/clock_offsets.jsonl \
      --neon-time-file "Neon Scene Camera v1 ps1.time" \
      --anchor-index 120=35 \
      --anchor-index 1700=984 \
      --output-csv motive_ubuntu_timestamps.csv

Automatically matching Motive head rotation against Neon IMU gyroscope data::

    python src/utils/sync_motive_neon.py \
      --motive-csv mocap.csv \
      --offset-log src/neon/time_sync/SESSION/clock_offsets.jsonl \
      --auto-align \
      --neon-imu-raw "NEON_RECORDING/imu/imu ps1.raw" \
      --neon-imu-time "NEON_RECORDING/imu/imu ps1.time" \
      --output-csv motive_ubuntu_timestamps.csv

The left side of every anchor is the value in the Motive CSV ``Frame`` column,
not Motive's global ``Capture Start Frame``.  Neon indices are zero based.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import datetime as dt
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


NS_PER_SECOND = 1_000_000_000
ISO_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<timezone>Z|[+-]\d{2}:\d{2})$"
)


@dataclass(frozen=True)
class MotiveExport:
    path: Path
    frame_rate_hz: float
    frame_ids: tuple[int, ...]
    first_frame: int
    capture_start_frame: int | None
    take_name: str | None


@dataclass(frozen=True)
class ClockAnchor:
    companion_ns: int
    ubuntu_ns: int
    stable: bool
    sequence: int | None


@dataclass(frozen=True)
class EventAnchor:
    motive_frame: int
    motive_elapsed_s: float
    companion_ns: int
    ubuntu_ns: int
    source: str


@dataclass(frozen=True)
class TimedSignal:
    time_s: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AutomaticAlignment:
    events: tuple[EventAnchor, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class AffineMap:
    """Ubuntu Unix nanoseconds = intercept_ns + slope * Motive elapsed ns."""

    slope: float
    intercept_ns: int
    residuals_ms: tuple[float, ...]

    def ubuntu_ns(self, motive_elapsed_s: float) -> int:
        return self.intercept_ns + round(
            self.slope * motive_elapsed_s * NS_PER_SECOND
        )


def normalized(value: str) -> str:
    return value.strip().lower()


def padded_cell(row: Sequence[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def metadata_pairs(rows: Sequence[Sequence[str]], before: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows[:before]:
        for index in range(0, len(row) - 1, 2):
            key = row[index].strip()
            if key:
                result[normalized(key)] = row[index + 1].strip()
    return result


def read_motive_csv(path: Path) -> MotiveExport:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"Motive CSV 为空：{path}")

    axis_rows = [
        index
        for index, row in enumerate(rows)
        if normalized(padded_cell(row, 0)) == "frame"
        and normalized(padded_cell(row, 1)) in {"time", "time (seconds)"}
    ]
    if len(axis_rows) != 1:
        raise ValueError(
            "无法唯一确定 Motive 的 Frame/Time 数据表头；"
            f"找到 {len(axis_rows)} 个候选"
        )
    axis_row = axis_rows[0]
    metadata = metadata_pairs(rows, axis_row)

    rate_text = metadata.get("export frame rate") or metadata.get(
        "capture frame rate"
    )
    if rate_text is None:
        raise ValueError("Motive CSV 缺少 Export/Capture Frame Rate")
    frame_rate = float(rate_text)
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"无效的 Motive 帧率：{rate_text!r}")

    frame_ids: list[int] = []
    for row_number, row in enumerate(rows[axis_row + 1 :], axis_row + 2):
        value = padded_cell(row, 0)
        if not value:
            continue
        try:
            frame_ids.append(int(value))
        except ValueError as exc:
            raise ValueError(
                f"Motive CSV 第 {row_number} 行 Frame 不是整数：{value!r}"
            ) from exc
    if not frame_ids:
        raise ValueError("Motive CSV 中没有数据帧")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Motive CSV 含重复 Frame 值")
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("Motive CSV 的 Frame 必须严格递增")

    capture_start: int | None = None
    capture_start_text = metadata.get("capture start frame")
    if capture_start_text:
        try:
            capture_start = int(capture_start_text)
        except ValueError:
            pass

    return MotiveExport(
        path=path,
        frame_rate_hz=frame_rate,
        frame_ids=tuple(frame_ids),
        first_frame=frame_ids[0],
        capture_start_frame=capture_start,
        take_name=metadata.get("take name") or None,
    )


def _find_header_row(
    rows: Sequence[Sequence[str]], label: str, before: int
) -> int:
    wanted = normalized(label)
    matches = [
        index
        for index, row in enumerate(rows[:before])
        if any(normalized(cell) == wanted for cell in row[:2])
    ]
    if not matches:
        raise ValueError(f"Motive CSV 缺少 {label!r} 表头行")
    return matches[-1]


def read_motive_head_angular_speed(
    path: Path,
    frame_rate_hz: float,
    preferred_name: str | None,
) -> tuple[TimedSignal, str]:
    """Read a head quaternion and return quaternion-difference angular speed."""

    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    axis_candidates = [
        index
        for index, row in enumerate(rows)
        if normalized(padded_cell(row, 0)) == "frame"
        and normalized(padded_cell(row, 1)) in {"time", "time (seconds)"}
    ]
    if len(axis_candidates) != 1:
        raise ValueError("无法确定 Motive Frame/Time 表头")
    axis_row = axis_candidates[0]
    type_row = _find_header_row(rows, "Type", axis_row)
    name_row = _find_header_row(rows, "Name", axis_row)

    quantity_candidates = [
        index
        for index in range(max(type_row, name_row) + 1, axis_row)
        if any(normalized(cell) == "rotation" for cell in rows[index])
    ]
    if not quantity_candidates:
        raise ValueError("Motive CSV 中没有 Rotation 表头")
    quantity_row = quantity_candidates[-1]

    width = max(
        len(rows[type_row]),
        len(rows[name_row]),
        len(rows[quantity_row]),
        len(rows[axis_row]),
    )
    rotation_axes: dict[str, dict[str, int]] = {}
    rotation_types: dict[str, str] = {}
    for column in range(width):
        if normalized(padded_cell(rows[quantity_row], column)) != "rotation":
            continue
        axis = padded_cell(rows[axis_row], column).upper()
        if axis not in {"X", "Y", "Z", "W"}:
            continue
        name = padded_cell(rows[name_row], column)
        if not name:
            continue
        rotation_axes.setdefault(name, {})[axis] = column
        rotation_types[name] = padded_cell(rows[type_row], column)

    complete = {
        name: axes for name, axes in rotation_axes.items() if set(axes) == {"X", "Y", "Z", "W"}
    }
    if preferred_name:
        exact = [name for name in complete if name == preferred_name]
        if not exact:
            exact = [name for name in complete if normalized(name) == normalized(preferred_name)]
        if len(exact) != 1:
            choices = ", ".join(sorted(complete)[:20])
            raise ValueError(
                f"找不到唯一的 --motive-head-name {preferred_name!r}；"
                f"可用 Rotation 名称包括：{choices}"
            )
        selected_name = exact[0]
    else:
        head_candidates = [
            name
            for name in complete
            if rotation_types.get(name) == "Bone"
            and normalized(name.split(":")[-1]) == "head"
        ]
        if len(head_candidates) != 1:
            choices = ", ".join(sorted(head_candidates or complete)[:20])
            raise ValueError(
                "无法自动唯一确定 Motive Head Rotation；请用 "
                f"--motive-head-name 指定。候选：{choices}"
            )
        selected_name = head_candidates[0]

    columns = complete[selected_name]
    quaternions: list[list[float]] = []
    frame_ids: list[int] = []
    for row in rows[axis_row + 1 :]:
        frame_text = padded_cell(row, 0)
        if not frame_text:
            continue
        frame_ids.append(int(frame_text))
        values: list[float] = []
        for axis in ("X", "Y", "Z", "W"):
            cell = padded_cell(row, columns[axis])
            values.append(float(cell) if cell else math.nan)
        quaternions.append(values)

    frame_array = np.asarray(frame_ids, dtype=np.float64)
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    valid = np.all(np.isfinite(quaternion_array), axis=1)
    norms = np.linalg.norm(quaternion_array, axis=1)
    valid &= norms > 1e-8
    quaternion_array[valid] /= norms[valid, None]

    left_valid = valid[:-1]
    right_valid = valid[1:]
    pair_valid = left_valid & right_valid
    frame_delta = np.diff(frame_array)
    pair_valid &= frame_delta > 0
    dots = np.full(len(frame_delta), np.nan, dtype=np.float64)
    dots[pair_valid] = np.abs(
        np.sum(
            quaternion_array[:-1][pair_valid]
            * quaternion_array[1:][pair_valid],
            axis=1,
        )
    )
    dots = np.clip(dots, 0.0, 1.0)
    delta_seconds = frame_delta / frame_rate_hz
    angular_speed = 2.0 * np.arccos(dots) / delta_seconds
    midpoint_s = (
        (frame_array[:-1] + frame_array[1:]) / 2.0 - frame_array[0]
    ) / frame_rate_hz
    finite = np.isfinite(angular_speed) & np.isfinite(midpoint_s)
    if np.count_nonzero(finite) < 20:
        raise ValueError(
            f"Motive Head {selected_name!r} 的有效旋转样本不足，无法自动同步"
        )
    return (
        TimedSignal(midpoint_s[finite], angular_speed[finite]),
        selected_name,
    )


def read_neon_imu_angular_speed(
    raw_path: Path, time_path: Path | None
) -> tuple[TimedSignal, np.ndarray, Path | None]:
    """Read a Neon raw IMU stream and return gyro magnitude and timestamps."""

    dtype_path = raw_path.with_name("imu.dtype")
    if dtype_path.exists():
        try:
            dtype_description = ast.literal_eval(dtype_path.read_text(encoding="utf-8"))
            dtype = np.dtype(dtype_description)
        except (OSError, SyntaxError, ValueError) as exc:
            raise ValueError(f"无法读取 Neon IMU dtype：{dtype_path}") from exc
    else:
        dtype = np.dtype(
            [
                ("timestamp_ns", "<i8"),
                ("gyro_x", "<f4"),
                ("gyro_y", "<f4"),
                ("gyro_z", "<f4"),
                ("accel_x", "<f4"),
                ("accel_y", "<f4"),
                ("accel_z", "<f4"),
                ("quaternion_w", "<f4"),
                ("quaternion_x", "<f4"),
                ("quaternion_y", "<f4"),
                ("quaternion_z", "<f4"),
            ]
        )
    required = {"gyro_x", "gyro_y", "gyro_z"}
    if dtype.names is None or not required.issubset(dtype.names):
        raise ValueError(f"Neon IMU dtype 缺少字段：{sorted(required)}")
    if raw_path.stat().st_size % dtype.itemsize:
        raise ValueError("Neon IMU raw 文件大小与 dtype 不匹配")
    samples = np.fromfile(raw_path, dtype=dtype)
    if len(samples) < 20:
        raise ValueError("Neon IMU 样本不足，无法自动同步")

    resolved_time_path = time_path
    if resolved_time_path is None:
        inferred = raw_path.with_suffix(".time")
        if inferred.exists():
            resolved_time_path = inferred
    if resolved_time_path is not None:
        timestamps = np.fromfile(resolved_time_path, dtype="<i8")
    elif "timestamp_ns" in dtype.names:
        timestamps = np.asarray(samples["timestamp_ns"], dtype=np.int64)
    else:
        raise ValueError(
            "IMU raw 中没有 timestamp_ns；请提供 --neon-imu-time"
        )
    if len(timestamps) != len(samples):
        raise ValueError(
            f"Neon IMU raw 有 {len(samples)} 个样本，但 time 有 {len(timestamps)} 个"
        )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Neon IMU 时间戳必须严格递增")

    gyro = np.column_stack(
        [samples["gyro_x"], samples["gyro_y"], samples["gyro_z"]]
    ).astype(np.float64)
    magnitude = np.linalg.norm(gyro, axis=1)
    relative_s = (timestamps - timestamps[0]).astype(np.float64) / NS_PER_SECOND
    finite = np.isfinite(magnitude)
    if np.count_nonzero(finite) < 20:
        raise ValueError("Neon IMU 陀螺仪有效样本不足")
    return (
        TimedSignal(relative_s[finite], magnitude[finite]),
        timestamps,
        resolved_time_path,
    )


def resample_signal(
    signal: TimedSignal, rate_hz: float, smoothing_seconds: float
) -> TimedSignal:
    if rate_hz <= 0 or not math.isfinite(rate_hz):
        raise ValueError("自动同步重采样率必须为正")
    order = np.argsort(signal.time_s)
    source_time = np.asarray(signal.time_s[order], dtype=np.float64)
    source_values = np.asarray(signal.values[order], dtype=np.float64)
    unique = np.concatenate(([True], np.diff(source_time) > 0))
    source_time = source_time[unique]
    source_values = source_values[unique]
    start = math.ceil(source_time[0] * rate_hz) / rate_hz
    stop = math.floor(source_time[-1] * rate_hz) / rate_hz
    if stop - start < 1.0:
        raise ValueError("用于自动同步的有效信号不足 1 秒")
    count = int(round((stop - start) * rate_hz)) + 1
    grid = start + np.arange(count, dtype=np.float64) / rate_hz
    values = np.interp(grid, source_time, source_values)

    window = max(1, round(smoothing_seconds * rate_hz))
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        values = np.convolve(values, kernel, mode="same")
    # Isolated optical-tracking spikes should not dominate the correlation.
    upper = float(np.percentile(values, 99.8))
    values = np.minimum(values, upper)
    return TimedSignal(grid, values)


def normalized_valid_correlation(long: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Pearson correlation for every placement of pattern fully inside long."""

    long = np.asarray(long, dtype=np.float64)
    pattern = np.asarray(pattern, dtype=np.float64)
    if len(pattern) < 10 or len(long) < len(pattern):
        raise ValueError("互相关信号长度不足")
    centered = pattern - np.mean(pattern)
    pattern_energy = float(np.dot(centered, centered))
    if pattern_energy <= 1e-12:
        raise ValueError("Motive 头部角速度几乎不变，无法自动同步")

    convolution_size = len(long) + len(pattern) - 1
    fft_size = 1 << (convolution_size - 1).bit_length()
    convolution = np.fft.irfft(
        np.fft.rfft(long, fft_size) * np.fft.rfft(centered[::-1], fft_size),
        fft_size,
    )[:convolution_size]
    numerator = convolution[len(pattern) - 1 : len(long)]

    cumulative = np.concatenate(([0.0], np.cumsum(long)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(long * long)))
    window_sum = cumulative[len(pattern) :] - cumulative[: -len(pattern)]
    window_sq = cumulative_sq[len(pattern) :] - cumulative_sq[: -len(pattern)]
    window_energy = window_sq - window_sum * window_sum / len(pattern)
    denominator = np.sqrt(np.maximum(window_energy, 0.0) * pattern_energy)
    correlation = np.full_like(numerator, -1.0)
    valid = denominator > 1e-12
    correlation[valid] = numerator[valid] / denominator[valid]
    return np.clip(correlation, -1.0, 1.0)


def correlation_peak(
    correlation: np.ndarray, exclusion_samples: int
) -> tuple[int, float, float]:
    best_index = int(np.argmax(correlation))
    best = float(correlation[best_index])
    mask = np.ones(len(correlation), dtype=bool)
    left = max(0, best_index - exclusion_samples)
    right = min(len(correlation), best_index + exclusion_samples + 1)
    mask[left:right] = False
    second = float(np.max(correlation[mask])) if np.any(mask) else -1.0
    return best_index, best, best - second


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
            "Neon IMU 有效时长短于 Motive；"
            "自动模式要求 Neon 覆盖完整 Motive 录制"
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
        search_signal = neon_uniform.values[
            search_left : search_right + window_count
        ]
        local_correlation = normalized_valid_correlation(search_signal, pattern)
        local_position, local_score, local_margin = correlation_peak(
            local_correlation, max(1, round(rate_hz * 0.15))
        )
        neon_position = search_left + local_position
        accepted = local_score >= min_correlation and local_margin >= min_peak_margin / 2
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
            motive_uniform.time_s[start]
            + (window_count - 1) / (2.0 * rate_hz)
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
                source=(
                    f"auto_imu:corr={local_score:.4f},margin={local_margin:.4f}"
                ),
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


def read_clock_anchors(path: Path, include_unstable: bool) -> list[ClockAnchor]:
    anchors: list[ClockAnchor] = []
    success_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"offset 日志第 {line_number} 行不是合法 JSON"
                ) from exc
            if record.get("status") != "ok":
                continue
            success_count += 1
            raw = record.get("interpolation_anchor")
            if not isinstance(raw, dict):
                continue
            quality = record.get("quality")
            stable = bool(
                isinstance(quality, dict)
                and quality.get("stable_for_interpolation", False)
            )
            if not include_unstable and not stable:
                continue
            try:
                companion_ns = int(raw["companion_unix_ns"])
                ubuntu_ns = int(raw["ubuntu_unix_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"offset 日志第 {line_number} 行的 interpolation_anchor 无效"
                ) from exc
            anchors.append(
                ClockAnchor(
                    companion_ns=companion_ns,
                    ubuntu_ns=ubuntu_ns,
                    stable=stable,
                    sequence=record.get("sequence"),
                )
            )

    if not anchors:
        detail = (
            "没有通过 stable_for_interpolation 检查的记录；若确认仍要使用，"
            "可添加 --include-unstable"
            if success_count and not include_unstable
            else "没有可用的成功记录"
        )
        raise ValueError(f"offset 日志中{detail}：{path}")

    anchors.sort(key=lambda item: item.companion_ns)
    deduplicated: list[ClockAnchor] = []
    for anchor in anchors:
        if deduplicated and anchor.companion_ns == deduplicated[-1].companion_ns:
            if anchor.ubuntu_ns != deduplicated[-1].ubuntu_ns:
                raise ValueError(
                    "offset 日志含同一 Companion 时间但 Ubuntu 时间不同的锚点"
                )
            continue
        if deduplicated and anchor.ubuntu_ns <= deduplicated[-1].ubuntu_ns:
            raise ValueError("offset 日志锚点不是单调时钟映射")
        deduplicated.append(anchor)
    return deduplicated


def _rounded_ratio(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest, with symmetric handling of signs."""

    if denominator <= 0:
        raise ValueError("插值分母必须为正")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def make_clock_converter(
    anchors: Sequence[ClockAnchor],
    *,
    source_attr: str,
    target_attr: str,
    allow_extrapolation: bool,
) -> Callable[[int], int]:
    source = [int(getattr(anchor, source_attr)) for anchor in anchors]
    target = [int(getattr(anchor, target_attr)) for anchor in anchors]

    def convert(timestamp_ns: int) -> int:
        position = bisect.bisect_left(source, timestamp_ns)
        if position < len(source) and source[position] == timestamp_ns:
            return target[position]
        if position == 0:
            if not allow_extrapolation:
                raise ValueError(
                    f"时间戳 {timestamp_ns} 早于 offset 日志覆盖范围 "
                    f"[{source[0]}, {source[-1]}]"
                )
            return timestamp_ns + (target[0] - source[0])
        if position == len(source):
            if not allow_extrapolation:
                raise ValueError(
                    f"时间戳 {timestamp_ns} 晚于 offset 日志覆盖范围 "
                    f"[{source[0]}, {source[-1]}]"
                )
            return timestamp_ns + (target[-1] - source[-1])

        source_left, source_right = source[position - 1], source[position]
        target_left, target_right = target[position - 1], target[position]
        delta = _rounded_ratio(
            (timestamp_ns - source_left) * (target_right - target_left),
            source_right - source_left,
        )
        return target_left + delta

    return convert


def parse_timestamp_ns(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("Neon 时间戳为空")
    try:
        if any(character in text for character in ".eE"):
            seconds = Decimal(text)
            if not seconds.is_finite() or seconds < 100_000_000:
                raise ValueError
            return int(
                (seconds * NS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_UP)
            )
        integer = int(text)
        if integer < 100_000_000_000_000_000:
            raise ValueError(
                "整数 Neon 时间戳应为 Unix 纳秒（通常为 19 位）；"
                "Unix 秒请写成带小数形式"
            )
        return integer
    except (ValueError, InvalidOperation):
        match = ISO_TIMESTAMP_RE.fullmatch(text)
        if match is not None:
            timezone = (
                "+00:00"
                if match.group("timezone") == "Z"
                else match.group("timezone")
            )
            whole_seconds = dt.datetime.fromisoformat(
                f"{match.group('date')}T{match.group('time')}{timezone}"
            )
            seconds_ns = int(whole_seconds.timestamp()) * NS_PER_SECOND
            fraction_ns = int((match.group("fraction") or "0").ljust(9, "0"))
            return seconds_ns + fraction_ns
        else:
            raise ValueError(
                f"无法解析 Neon 时间戳 {value!r}；请使用 Unix 纳秒、"
                "带小数的 Unix 秒或带时区 ISO-8601"
            ) from None


def parse_pair(value: str, right_parser: Callable[[str], int]) -> tuple[int, int]:
    if "=" not in value:
        raise ValueError(f"锚点 {value!r} 应使用 MOTIVE_FRAME=NEON_VALUE 格式")
    left, right = value.split("=", 1)
    try:
        motive_frame = int(left.strip())
    except ValueError as exc:
        raise ValueError(f"Motive frame 不是整数：{left!r}") from exc
    return motive_frame, right_parser(right.strip())


def read_neon_time_file(path: Path) -> tuple[int, ...]:
    size = path.stat().st_size
    if size % 8:
        raise ValueError(f"Neon .time 文件大小不是 8 字节的整数倍：{path}")
    timestamps: list[int] = []
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 65536):
            count = len(chunk) // 8
            timestamps.extend(struct.unpack(f"<{count}q", chunk))
    if not timestamps:
        raise ValueError(f"Neon .time 文件为空：{path}")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Neon .time 时间戳不是单调递增：{path}")
    return tuple(timestamps)


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
                motive_elapsed_s=(frame - motive.first_frame)
                / motive.frame_rate_hz,
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
        intercept = event.ubuntu_ns - round(
            event.motive_elapsed_s * NS_PER_SECOND
        )
        return AffineMap(slope=1.0, intercept_ns=intercept, residuals_ms=(0.0,))

    mean_x = sum(event.motive_elapsed_s for event in events) / len(events)
    reference_ns = events[0].ubuntu_ns
    relative_y = [
        (event.ubuntu_ns - reference_ns) / NS_PER_SECOND for event in events
    ]
    mean_y = sum(relative_y) / len(relative_y)
    denominator = sum((event.motive_elapsed_s - mean_x) ** 2 for event in events)
    if denominator == 0:
        raise ValueError("同步锚点没有 Motive 时间跨度，无法拟合")
    slope = sum(
        (event.motive_elapsed_s - mean_x) * (value - mean_y)
        for event, value in zip(events, relative_y)
    ) / denominator
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError(f"拟合得到无效的时钟比例：{slope}")

    # Keep the 19-digit Unix timestamp out of floating-point arithmetic.
    intercept = reference_ns + round(
        (mean_y - slope * mean_x) * NS_PER_SECOND
    )
    model = AffineMap(slope=slope, intercept_ns=intercept, residuals_ms=())
    residuals = tuple(
        (event.ubuntu_ns - model.ubuntu_ns(event.motive_elapsed_s)) / 1_000_000
        for event in events
    )
    return AffineMap(slope=slope, intercept_ns=intercept, residuals_ms=residuals)


def utc_iso(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, NS_PER_SECOND)
    base = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    return f"{base:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def write_outputs(
    output_csv: Path,
    report_path: Path,
    motive: MotiveExport,
    clock_anchors: Sequence[ClockAnchor],
    events: Sequence[EventAnchor],
    model: AffineMap,
    ubuntu_to_companion: Callable[[int], int],
    args: argparse.Namespace,
    automatic_diagnostics: dict[str, object] | None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    first_ubuntu = model.ubuntu_ns(0.0)
    last_elapsed = (motive.frame_ids[-1] - motive.first_frame) / motive.frame_rate_hz
    last_ubuntu = model.ubuntu_ns(last_elapsed)

    # Validate clock-log coverage before creating a partial CSV.
    first_companion = ubuntu_to_companion(first_ubuntu)
    last_companion = ubuntu_to_companion(last_ubuntu)

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "motive_frame",
                "motive_elapsed_s",
                "ubuntu_unix_ns",
                "ubuntu_utc_iso",
                "estimated_neon_companion_unix_ns",
            ]
        )
        for frame in motive.frame_ids:
            elapsed_s = (frame - motive.first_frame) / motive.frame_rate_hz
            ubuntu_ns = model.ubuntu_ns(elapsed_s)
            companion_ns = ubuntu_to_companion(ubuntu_ns)
            writer.writerow(
                [
                    frame,
                    f"{elapsed_s:.9f}",
                    ubuntu_ns,
                    utc_iso(ubuntu_ns),
                    companion_ns,
                ]
            )

    residual_abs = [abs(value) for value in model.residuals_ms]
    report = {
        "schema_version": 1,
        "purpose": "Offline Motive to Ubuntu/Neon clock synchronization",
        "inputs": {
            "motive_csv": str(motive.path.resolve()),
            "clock_offsets_jsonl": str(args.offset_log.resolve()),
            "neon_time_file": (
                str(args.neon_time_file.resolve()) if args.neon_time_file else None
            ),
            "neon_imu_raw": (
                str(args.neon_imu_raw.resolve()) if args.neon_imu_raw else None
            ),
            "neon_imu_time": (
                str(args.neon_imu_time.resolve()) if args.neon_imu_time else None
            ),
        },
        "motive": {
            "take_name": motive.take_name,
            "frame_rate_hz": motive.frame_rate_hz,
            "first_exported_frame": motive.first_frame,
            "last_exported_frame": motive.frame_ids[-1],
            "frame_count": len(motive.frame_ids),
            "capture_start_frame_metadata": motive.capture_start_frame,
        },
        "time_echo": {
            "used_anchor_count": len(clock_anchors),
            "included_unstable_records": bool(args.include_unstable),
            "allowed_out_of_range_endpoint_offset": bool(
                args.allow_offset_extrapolation
            ),
            "companion_coverage_ns": [
                clock_anchors[0].companion_ns,
                clock_anchors[-1].companion_ns,
            ],
            "ubuntu_coverage_ns": [
                clock_anchors[0].ubuntu_ns,
                clock_anchors[-1].ubuntu_ns,
            ],
        },
        "event_anchors": [
            {
                "motive_frame": event.motive_frame,
                "motive_elapsed_s": event.motive_elapsed_s,
                "neon_companion_unix_ns": event.companion_ns,
                "ubuntu_unix_ns_after_time_echo": event.ubuntu_ns,
                "source": event.source,
                "fit_residual_ms": residual,
            }
            for event, residual in zip(events, model.residuals_ms)
        ],
        "automatic_alignment": automatic_diagnostics,
        "fitted_mapping": {
            "formula": (
                "ubuntu_ns = intercept_ns + slope * motive_elapsed_s * 1e9"
            ),
            "slope": model.slope,
            "estimated_rate_drift_ppm": (model.slope - 1.0) * 1_000_000,
            "intercept_ns": model.intercept_ns,
            "one_anchor_assumes_zero_drift": len(events) == 1,
            "max_absolute_anchor_residual_ms": max(residual_abs),
            "rms_anchor_residual_ms": math.sqrt(
                sum(value * value for value in model.residuals_ms)
                / len(model.residuals_ms)
            ),
        },
        "mapped_range": {
            "first_ubuntu_unix_ns": first_ubuntu,
            "last_ubuntu_unix_ns": last_ubuntu,
            "first_ubuntu_utc": utc_iso(first_ubuntu),
            "last_ubuntu_utc": utc_iso(last_ubuntu),
            "first_estimated_neon_companion_unix_ns": first_companion,
            "last_estimated_neon_companion_unix_ns": last_companion,
        },
        "outputs": {"frame_timestamps_csv": str(output_csv.resolve())},
        "limitations": [
            "Accuracy is limited by how precisely the same physical event is selected in both recordings.",
            "Time Echo has millisecond timestamp resolution and does not guarantee sub-millisecond absolute accuracy.",
            "With one event anchor, clock-rate drift cannot be estimated.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "用共同事件和 Neon Time Echo 日志，将 Motive 导出帧映射到 "
            "Ubuntu/Neon Companion 时间轴"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motive-csv", type=Path, required=True, help="Motive CSV")
    parser.add_argument(
        "--offset-log",
        type=Path,
        required=True,
        help="clock_offset_monitor.py 生成的 clock_offsets.jsonl",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        metavar="MOTIVE_FRAME=NEON_TIMESTAMP",
        help=(
            "共同事件；Neon 时间可为 Unix 纳秒、带小数 Unix 秒或带时区 ISO。"
            "可重复；两个以上锚点可估计漂移"
        ),
    )
    parser.add_argument(
        "--neon-time-file",
        type=Path,
        help="Neon 导出的二进制 *.time 文件（little-endian int64 Unix ns）",
    )
    parser.add_argument(
        "--anchor-index",
        action="append",
        default=[],
        metavar="MOTIVE_FRAME=NEON_INDEX",
        help="共同事件在 Neon .time 中的零基索引；可重复",
    )
    parser.add_argument(
        "--auto-align",
        action="store_true",
        help="自动匹配 Motive 头部旋转角速度与 Neon IMU 陀螺仪",
    )
    parser.add_argument(
        "--neon-imu-raw",
        type=Path,
        help="自动模式使用的 Neon 'imu ps1.raw'",
    )
    parser.add_argument(
        "--neon-imu-time",
        type=Path,
        help="对应的 'imu ps1.time'；省略时从 raw 文件名自动推断",
    )
    parser.add_argument(
        "--motive-head-name",
        help="Motive Rotation 对象全名；默认自动寻找名称以 ':Head' 结尾的 Bone",
    )
    parser.add_argument(
        "--auto-rate",
        type=float,
        default=100.0,
        help="自动互相关的重采样频率 Hz",
    )
    parser.add_argument(
        "--auto-smoothing-seconds",
        type=float,
        default=0.05,
        help="自动匹配角速度的移动平均窗口",
    )
    parser.add_argument(
        "--auto-window-seconds",
        type=float,
        default=6.0,
        help="用于生成自动局部锚点的窗口长度",
    )
    parser.add_argument(
        "--auto-local-search-seconds",
        type=float,
        default=0.75,
        help="局部锚点在全局匹配附近的搜索半径",
    )
    parser.add_argument(
        "--auto-min-correlation",
        type=float,
        default=0.40,
        help="接受自动匹配所需的最低相关系数",
    )
    parser.add_argument(
        "--auto-min-peak-margin",
        type=float,
        default=0.03,
        help="最佳相关峰相对次优峰的最低优势",
    )
    parser.add_argument(
        "--include-unstable",
        action="store_true",
        help="也使用未通过 Time Echo stable_for_interpolation 检查的记录",
    )
    parser.add_argument(
        "--allow-offset-extrapolation",
        action="store_true",
        help=(
            "允许在 Time Echo 覆盖范围之外使用最近端点 offset；"
            "风险高，正常录制不建议"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="逐 Motive 帧时间戳输出 CSV",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="同步拟合报告；默认与 output CSV 同目录、同名并加 _report.json",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    output_csv = args.output_csv.resolve()
    report_path = (
        args.report_json.resolve()
        if args.report_json
        else output_csv.with_name(f"{output_csv.stem}_report.json")
    )
    if output_csv == report_path:
        raise ValueError("--output-csv 与 --report-json 必须是不同文件")
    input_paths = [args.motive_csv.resolve(), args.offset_log.resolve()]
    for optional_path in (
        args.neon_time_file,
        args.neon_imu_raw,
        args.neon_imu_time,
    ):
        if optional_path is not None:
            input_paths.append(optional_path.resolve())
    if output_csv in input_paths:
        raise ValueError("--output-csv 不能覆盖任何输入文件")
    if report_path in input_paths:
        raise ValueError("--report-json 不能覆盖任何输入文件")
    if args.auto_align and (args.anchor or args.anchor_index):
        raise ValueError("--auto-align 不能与手工 --anchor/--anchor-index 同时使用")
    if args.auto_align and args.neon_imu_raw is None:
        raise ValueError("--auto-align 必须提供 --neon-imu-raw")
    if not 0.0 < args.auto_min_correlation <= 1.0:
        raise ValueError("--auto-min-correlation 必须在 (0, 1] 内")
    if not 0.0 <= args.auto_min_peak_margin <= 2.0:
        raise ValueError("--auto-min-peak-margin 必须在 [0, 2] 内")
    if args.auto_window_seconds <= 0 or args.auto_local_search_seconds <= 0:
        raise ValueError("自动同步窗口和搜索时间必须为正")
    if args.auto_smoothing_seconds < 0:
        raise ValueError("--auto-smoothing-seconds 不能为负")

    motive = read_motive_csv(args.motive_csv)
    clock_anchors = read_clock_anchors(args.offset_log, args.include_unstable)
    companion_to_ubuntu = make_clock_converter(
        clock_anchors,
        source_attr="companion_ns",
        target_attr="ubuntu_ns",
        allow_extrapolation=args.allow_offset_extrapolation,
    )
    ubuntu_to_companion = make_clock_converter(
        clock_anchors,
        source_attr="ubuntu_ns",
        target_attr="companion_ns",
        allow_extrapolation=args.allow_offset_extrapolation,
    )
    automatic_diagnostics: dict[str, object] | None = None
    if args.auto_align:
        motive_signal, selected_head = read_motive_head_angular_speed(
            args.motive_csv,
            motive.frame_rate_hz,
            args.motive_head_name,
        )
        neon_signal, neon_timestamps, resolved_imu_time = read_neon_imu_angular_speed(
            args.neon_imu_raw,
            args.neon_imu_time,
        )
        automatic = automatic_event_anchors(
            motive,
            motive_signal,
            neon_signal,
            int(neon_timestamps[0]),
            companion_to_ubuntu,
            rate_hz=args.auto_rate,
            smoothing_seconds=args.auto_smoothing_seconds,
            window_seconds=args.auto_window_seconds,
            local_search_seconds=args.auto_local_search_seconds,
            min_correlation=args.auto_min_correlation,
            min_peak_margin=args.auto_min_peak_margin,
        )
        events = list(automatic.events)
        automatic_diagnostics = {
            **automatic.diagnostics,
            "selected_motive_rotation": selected_head,
            "resolved_neon_imu_time": (
                str(resolved_imu_time.resolve()) if resolved_imu_time else None
            ),
        }
    else:
        neon_times = (
            read_neon_time_file(args.neon_time_file)
            if args.neon_time_file
            else None
        )
        events = build_event_anchors(
            motive,
            args.anchor,
            args.anchor_index,
            neon_times,
            companion_to_ubuntu,
        )
    model = fit_affine_map(events)
    write_outputs(
        output_csv,
        report_path,
        motive,
        clock_anchors,
        events,
        model,
        ubuntu_to_companion,
        args,
        automatic_diagnostics,
    )

    max_residual = max(abs(value) for value in model.residuals_ms)
    print(f"完成：{len(motive.frame_ids)} 个 Motive 帧")
    print(
        f"映射：ubuntu_ns = {model.intercept_ns} + "
        f"{model.slope:.12f} * motive_elapsed_s * 1e9"
    )
    print(f"估计速率漂移：{(model.slope - 1.0) * 1e6:+.3f} ppm")
    print(f"同步锚点最大残差：{max_residual:.3f} ms")
    if automatic_diagnostics is not None:
        print(
            "自动匹配："
            f"global_corr={automatic_diagnostics['global_correlation']:.3f}, "
            f"peak_margin={automatic_diagnostics['global_peak_margin']:.3f}, "
            f"anchors={len(events)}"
        )
    print(f"逐帧时间戳：{output_csv}")
    print(f"同步报告：{report_path}")
    if len(events) == 1:
        print("注意：只有一个同步锚点，已假定 Motive 与 Ubuntu 无时钟速率漂移。")
    if abs((model.slope - 1.0) * 1e6) > 1_000:
        print(
            "警告：估计漂移超过 1000 ppm；请检查共同事件的帧选择是否准确。",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
