"""Load a local neutral SMPL-X model and visually retarget solved Motive bones.

This uses neutral shape and linear blend skinning, not MoSh++ shape/pose fitting.
Untracked legs and face retain their neutral pose relative to their parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.avatars.pose import anatomical_frames, limb_frames
from utils.smplx.models import SMPLX_TO_MUJOCO

# Official SMPL-X vertex landmark: https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py
NOSE_VERTEX = 9120


@dataclass
class NeutralAvatar:
    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    joints: np.ndarray
    parents: np.ndarray
    nose_bridge: np.ndarray


def load_neutral_model(path: Path) -> NeutralAvatar:
    """Read a licensed NPZ model with NumPy only; no model data is distributed."""
    if Path(path).suffix.lower() != ".npz":
        raise ValueError(
            "Use the official SMPLX_NEUTRAL.npz model (pickle is not loaded)"
        )
    with np.load(path, allow_pickle=False) as model:
        if (
            "gender" in model
            and str(np.asarray(model["gender"]).item()).lower() != "neutral"
        ):
            raise ValueError("This viewer expects a neutral-gender SMPL-X model")
        vertices = np.asarray(model["v_template"], dtype=float)
        faces = np.asarray(model["f"], dtype=int)
        weights = np.asarray(model["weights"], dtype=float)
        regressor = np.asarray(model["J_regressor"], dtype=float)
        parents = np.asarray(model["kintree_table"][0], dtype=np.int64).copy()
    if (
        vertices.shape != (10475, 3)
        or weights.shape != (10475, 55)
        or regressor.shape != (55, 10475)
    ):
        raise ValueError("Expected the standard 10475-vertex, 55-joint SMPL-X topology")
    if not np.isfinite(vertices).all() or not np.isfinite(weights).all():
        raise ValueError("Model contains nonfinite data")
    if not np.allclose(weights.sum(axis=1), 1, atol=2e-5):
        raise ValueError("SMPL-X skin weights do not sum to one")
    joints = regressor @ vertices
    parents[0] = -1
    for index, parent in enumerate(parents[1:], 1):
        if not 0 <= parent < index:
            raise ValueError("SMPL-X joint parents are not in topological order")
    # Nose-tip landmark plus a provisional bridge offset (up/back in SMPL-X).
    # This is an editable visual placement, never a measured head-camera extrinsic.
    bridge = vertices[NOSE_VERTEX] + [0, 0.015, -0.012]
    transform = SMPLX_TO_MUJOCO.T
    return NeutralAvatar(
        vertices @ transform,
        faces,
        weights,
        joints @ transform,
        parents,
        bridge @ transform,
    )


def retarget_neutral(
    avatar: NeutralAvatar,
    bones: dict[str, np.ndarray],
    head_top: np.ndarray,
    head_front: np.ndarray,
    reference: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return joint positions, global deformation matrices, and body scale."""
    if not 0 <= reference < len(head_top):
        raise ValueError("Retarget reference is outside the recording")
    mapping = {
        0: next(iter(bones)),
        3: "Ab",
        6: "Ab",
        9: "Chest",
        12: "Neck",
        13: "LShoulder",
        14: "RShoulder",
        15: "Head",
        16: "LUArm",
        17: "RUArm",
        18: "LFArm",
        19: "RFArm",
        20: "LHand",
        21: "RHand",
    }
    for start, side in ((25, "L"), (40, "R")):
        for finger_index, finger in enumerate(
            ("Index", "Middle", "Pinky", "Ring", "Thumb")
        ):
            for segment in range(3):
                mapping[start + finger_index * 3 + segment] = (
                    f"{side}{finger}{segment + 1}"
                )
    missing = sorted(set(mapping.values()).difference(bones))
    if missing:
        raise ValueError(f"Missing solved bones for SMPL-X retargeting: {missing}")
    source = avatar.joints
    target = {index: bones[name][:, 4:7] for index, name in mapping.items()}
    rest_up = source[15] - source[0]
    up = target[15] - target[0]
    source_body = anatomical_frames(source[13] - source[14], rest_up)[0]
    target_body = anatomical_frames(target[13] - target[14], up)
    alignment = {index: target_body[reference] @ source_body.T for index in mapping}
    alignment[15] = (
        anatomical_frames(
            target[13] - target[14], head_top - target[15], head_front - target[15]
        )[reference]
        @ source_body.T
    )
    for parent, child in ((13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)):
        rest_frame = limb_frames(source[child] - source[parent], rest_up)[0]
        target_frame = limb_frames(target[child] - target[parent], up)[reference]
        alignment[parent] = target_frame @ rest_frame.T
    for start, wrist in ((25, 20), (40, 21)):
        for finger in range(5):
            first = start + finger * 3
            for segment in range(3):
                joint = first + segment
                a, b = (joint, joint + 1) if segment < 2 else (joint - 1, joint)
                rest_frame = limb_frames(source[b] - source[a], rest_up)[0]
                target_frame = limb_frames(target[b] - target[a], up)[reference]
                alignment[joint] = target_frame @ rest_frame.T
        rest_frame = limb_frames(source[start + 3] - source[wrist], rest_up)[0]
        target_frame = limb_frames(target[start + 3] - target[wrist], up)[reference]
        alignment[wrist] = target_frame @ rest_frame.T
    frames = len(head_top)
    rotations = np.empty((frames, 55, 3, 3))
    for joint in range(55):
        if joint in mapping:
            solved = Rotation.from_quat(bones[mapping[joint]][:, :4]).as_matrix()
            rotations[:, joint] = solved @ solved[reference].T @ alignment[joint]
        else:
            rotations[:, joint] = rotations[:, avatar.parents[joint]]
    height = float(avatar.vertices[:, 2].max() - source[0, 2])
    scale = float(
        np.clip(
            np.median(np.linalg.norm(head_top - target[0], axis=1)) / height, 0.6, 1.4
        )
    )
    positions = np.empty((frames, 55, 3))
    positions[:, 0] = target[0]
    for joint in range(1, 55):
        parent = avatar.parents[joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "fij,j->fi", rotations[:, parent], scale * (source[joint] - source[parent])
        )
    return positions, rotations, scale
