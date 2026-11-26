"""Reusable synchronization projection helpers."""

from __future__ import annotations

import numpy as np

from utils.synchronization.motive_motion import (
    normalized,
)
from utils.synchronization.video_models import (
    MotiveMotion,
    Projection,
)


def nearest_index(sorted_values: np.ndarray, target: int) -> int:
    position = int(np.searchsorted(sorted_values, target, side="left"))
    if position <= 0:
        return 0
    if position >= len(sorted_values):
        return len(sorted_values) - 1
    before = int(sorted_values[position - 1])
    after = int(sorted_values[position])
    return position - 1 if target - before <= after - target else position


def projected_coordinates(positions: np.ndarray) -> np.ndarray:
    """Fixed isometric projection for Motive's usual Y-up coordinates."""

    result = np.empty((*positions.shape[:-1], 2), dtype=np.float64)
    result[..., 0] = positions[..., 0] - 0.38 * positions[..., 2]
    result[..., 1] = positions[..., 1] + 0.16 * positions[..., 2]
    return result


def make_projection(motion: MotiveMotion, width: int, height: int) -> Projection:
    projected = projected_coordinates(motion.positions)
    finite = np.isfinite(projected).all(axis=2)
    minimum_points = 2 if motion.visualization_mode == "rigid-body" else 10
    if np.count_nonzero(finite) < minimum_points:
        raise ValueError("Motive 有效位置不足，无法可视化")
    values = projected[finite]
    low = np.percentile(values, 1.0, axis=0)
    high = np.percentile(values, 99.0, axis=0)
    minimum_span = 0.5 if normalized(motion.source_units) in {"m", "meters"} else 500.0
    span = np.maximum(high - low, minimum_span)
    drawable_width = width * 0.82
    drawable_height = height * 0.62
    scale = min(drawable_width / span[0], drawable_height / span[1])
    return Projection(center=(low + high) / 2.0, scale=float(scale))


def bone_color(local_name: str) -> tuple[int, int, int]:
    if local_name.startswith("L"):
        return (255, 180, 55)
    if local_name.startswith("R"):
        return (70, 150, 255)
    return (80, 230, 130)


def project_to_pixels(
    world_points: np.ndarray,
    projection: Projection,
    width: int,
    height: int,
) -> np.ndarray:
    projected = projected_coordinates(np.asarray(world_points, dtype=np.float64))
    pixels = np.empty_like(projected)
    pixels[..., 0] = (
        width / 2.0 + (projected[..., 0] - projection.center[0]) * projection.scale
    )
    pixels[..., 1] = (
        height * 0.60 - (projected[..., 1] - projection.center[1]) * projection.scale
    )
    return pixels


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray | None:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        return None
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return None
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def unit_scale_to_meters(units: str) -> float:
    scales = {
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
    return scales.get(normalized(units), 1.0)
