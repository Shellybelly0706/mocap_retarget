"""Reusable smplx models helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# SMPL-X is Y-up.  MuJoCo scenes conventionally use Z-up.  For row vectors,
# points are converted with p_mj = p_smplx @ SMPLX_TO_MUJOCO.T.
SMPLX_TO_MUJOCO = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


REQUIRED_RESULT_KEYS = ("trans", "root_orient", "pose_body", "pose_hand", "betas")


REQUIRED_MODEL_KEYS = (
    "v_template",
    "f",
    "weights",
    "J_regressor",
    "kintree_table",
    "shapedirs",
)


FINGER_LABEL_FALLBACKS = {
    "RIndex": "CustomMarker6",
    "RMiddle": "CustomMarker4",
    "RHandIn": "CustomMarker1",
}


@dataclass(frozen=True)
class StaticScene:
    table_center_xy: np.ndarray
    table_axis_x: np.ndarray
    table_axis_y: np.ndarray
    table_top_z: float
    table_length: float
    table_width: float
    table_thickness: float
    cup_center: np.ndarray
    cup_positions: np.ndarray
    cup_quaternions: np.ndarray
    cup_height: float
    cup_radius: float
    index_contact: np.ndarray
    middle_contact: np.ndarray
    contact_frame: int
    pickup_frame: int
    grip_frame: int
    cup_is_static: bool
