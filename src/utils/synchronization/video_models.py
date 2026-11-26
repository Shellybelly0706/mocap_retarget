"""Reusable synchronization video models helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotiveMotion:
    path: Path
    take_name: str
    capture_start_text: str
    capture_start_ubuntu_ns: int
    frame_rate_hz: float
    frame_ids: np.ndarray
    ubuntu_ns: np.ndarray
    visualization_mode: str
    object_name: str
    skeleton_namespace: str
    bone_names: tuple[str, ...]
    positions: np.ndarray
    quaternions_xyzw: np.ndarray | None
    edges: tuple[tuple[int, int], ...]
    source_units: str


@dataclass(frozen=True)
class NeonVideo:
    recording_dir: Path
    video_path: Path
    time_path: Path | None
    recording_start_companion_ns: int
    recording_start_ubuntu_ns: int
    companion_ns: np.ndarray
    ubuntu_ns: np.ndarray
    fps: float
    width: int
    height: int
    recording_name: str


@dataclass(frozen=True)
class Projection:
    center: np.ndarray
    scale: float
