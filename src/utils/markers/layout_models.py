"""Reusable markers layout models helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

# This editor stores markers strictly on the SMPL-X surface.  A positive
# MoSh++ distance_from_skin would move them outside the mesh along its normals;
# a negative value would move them inside it.
DEFAULT_DISTANCE_FROM_SKIN_M = 0.0


# Matplotlib's 3-D depth sorting can hide a scatter point that is exactly
# coplanar with an opaque triangle.  This offset is display-only; saved marker
# vertex IDs and MoSh++'s distance_from_skin remain exactly on the surface.
DISPLAY_MARKER_OFFSET_M = 0.006


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    path: Path
    sha256: str


@dataclass(frozen=True)
class ValidationResult:
    labels: tuple[str, ...]
    assigned: tuple[str, ...]
    unassigned: tuple[str, ...]
    duplicate_vertices: Mapping[int, tuple[str, ...]]
    vertex_count: int | None
