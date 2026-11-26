"""Reusable avatars scene xml helpers."""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

from utils.avatars.pose import (
    matrices_to_mujoco_quaternions,
)


def static_box_xml(
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    axes: np.ndarray,
    rgba: str | None,
    material: str | None = None,
) -> str:
    """Return one oriented, world-fixed MuJoCo box geom."""
    if (rgba is None) == (material is None):
        raise ValueError("A static box requires exactly one of rgba or material")
    quaternion = matrices_to_mujoco_quaternions(np.asarray(axes))
    appearance = (
        f'rgba="{rgba}"' if rgba is not None else f'material="{escape(material)}"'
    )
    return (
        f'<geom name="{escape(name)}" type="box" '
        f'pos="{xml_values(center)}" size="{xml_values(half_size)}" '
        f'quat="{xml_values(quaternion)}" {appearance}/>'
    )


def xml_values(values: np.ndarray, precision: int = 7) -> str:
    return " ".join(f"{value:.{precision}g}" for value in values.reshape(-1))
