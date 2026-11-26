"""Reusable synchronization signals helpers."""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path

import numpy as np

from utils.synchronization.constants import (
    NS_PER_SECOND,
)
from utils.synchronization.models import (
    TimedSignal,
)
from utils.synchronization.motive import (
    _find_header_row,
    normalized,
    padded_cell,
)


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
        name: axes
        for name, axes in rotation_axes.items()
        if set(axes) == {"X", "Y", "Z", "W"}
    }
    if preferred_name:
        exact = [name for name in complete if name == preferred_name]
        if not exact:
            exact = [
                name
                for name in complete
                if normalized(name) == normalized(preferred_name)
            ]
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
            quaternion_array[:-1][pair_valid] * quaternion_array[1:][pair_valid],
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
        raise ValueError("IMU raw 中没有 timestamp_ns；请提供 --neon-imu-time")
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
