"""Reusable avatars models helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AnatomicalPart:
    name: str
    geom_type: str
    size: tuple[float, ...]
    material: str
    positions: np.ndarray
    quaternions: np.ndarray


@dataclass(frozen=True)
class AvatarAsset:
    vertices: np.ndarray
    texcoords: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    joint_names: list[str]
    bind_positions: np.ndarray
    bind_rotations: np.ndarray


@dataclass(frozen=True)
class AvatarConfig:
    name: str
    source: Path
    skin: Path
    texture: Path


@dataclass(frozen=True)
class SceneObstacles:
    """Static room geometry inferred from the start and end calibration poses."""

    xml: str
    table_corner: np.ndarray
    table_length: float
    board_corner: np.ndarray
    table_forward: np.ndarray
    wall_normal: np.ndarray
    wall_contact: np.ndarray
    wall_span: float
    measured_board_corner_height: float
    cylinder_positions: np.ndarray
    cylinder_pickup_frames: tuple[int, int]
    cylinder_release_frames: tuple[int, int]
