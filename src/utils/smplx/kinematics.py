"""Reusable smplx kinematics helpers."""

from __future__ import annotations

import numpy as np

from utils.smplx.models import (
    SMPLX_TO_MUJOCO,
)


def require_keys(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"{name} is missing required arrays: {', '.join(missing)}")


def axis_angle_to_matrix(rotvecs: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion for arrays ending in three values."""
    vectors = np.asarray(rotvecs, dtype=np.float64)
    original_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, 3)
    angles = np.linalg.norm(flat, axis=1)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None], len(flat), axis=0)

    active = angles > 1e-12
    if np.any(active):
        axes = flat[active] / angles[active, None]
        x, y, z = axes.T
        zeros = np.zeros_like(x)
        skew = np.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), axis=1).reshape(
            -1, 3, 3
        )
        sine = np.sin(angles[active])[:, None, None]
        cosine = np.cos(angles[active])[:, None, None]
        identity = np.eye(3, dtype=np.float64)[None]
        matrices[active] = identity + sine * skew + (1.0 - cosine) * (skew @ skew)

    return matrices.reshape(*original_shape, 3, 3)


def matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to MuJoCo quaternions in w, x, y, z order."""
    import mujoco

    matrices = np.asarray(matrices, dtype=np.float64)
    quaternions = np.empty((*matrices.shape[:-2], 4), dtype=np.float64)
    flat_matrices = matrices.reshape(-1, 3, 3)
    flat_quaternions = quaternions.reshape(-1, 4)
    for matrix, quaternion in zip(flat_matrices, flat_quaternions, strict=True):
        mujoco.mju_mat2Quat(quaternion, matrix.reshape(9))
    return quaternions


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    import mujoco

    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def skinned_vertices_at_frame(
    vertices: np.ndarray,
    bind_positions: np.ndarray,
    weights: np.ndarray,
    joint_positions: np.ndarray,
    joint_quaternions: np.ndarray,
    frame: int,
) -> np.ndarray:
    """Evaluate the same linear-blend skin used by MuJoCo for one frame."""
    output = np.zeros_like(vertices, dtype=np.float64)
    for joint in range(len(bind_positions)):
        active = weights[:, joint] > 1e-8
        if not np.any(active):
            continue
        rotation = quaternion_to_matrix(joint_quaternions[frame, joint])
        transformed = (
            vertices[active] - bind_positions[joint]
        ) @ rotation.T + joint_positions[frame, joint]
        output[active] += weights[active, joint, None] * transformed
    return output


def full_pose(result: np.lib.npyio.NpzFile) -> np.ndarray:
    """Assemble poses in the same 55-joint order used by BodyModel."""
    frames = len(result["trans"])

    def value(name: str, width: int) -> np.ndarray:
        if name not in result.files:
            return np.zeros((frames, width), dtype=np.float64)
        array = np.asarray(result[name], dtype=np.float64)
        if array.shape != (frames, width):
            raise ValueError(
                f"{name} must have shape {(frames, width)}, got {array.shape}"
            )
        return array

    pose = np.concatenate(
        (
            value("root_orient", 3),
            value("pose_body", 63),
            value("pose_jaw", 3),
            value("pose_eye", 6),
            value("pose_hand", 90),
        ),
        axis=1,
    )
    return pose.reshape(frames, 55, 3)


def shaped_template_and_joints(
    result: np.lib.npyio.NpzFile, model: np.lib.npyio.NpzFile
) -> tuple[np.ndarray, np.ndarray]:
    betas = np.asarray(result["betas"], dtype=np.float64).reshape(-1)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    if len(betas) > shapedirs.shape[2]:
        raise ValueError(
            f"Result has {len(betas)} betas, but model only has "
            f"{shapedirs.shape[2]} shape directions"
        )
    vertices = np.asarray(model["v_template"], dtype=np.float64).copy()
    vertices += np.einsum("vcn,n->vc", shapedirs[:, :, : len(betas)], betas)
    regressor = np.asarray(model["J_regressor"], dtype=np.float64)
    joints = regressor @ vertices
    return vertices, joints


def global_joint_motion(
    poses: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global joint positions and rotations in SMPL-X coordinates."""
    local_rotations = axis_angle_to_matrix(poses)
    frames, joint_count = local_rotations.shape[:2]
    if len(rest_joints) != joint_count or len(parents) != joint_count:
        raise ValueError(
            f"Pose/model joint mismatch: pose={joint_count}, "
            f"joints={len(rest_joints)}, parents={len(parents)}"
        )

    rotations = np.empty_like(local_rotations)
    positions = np.empty((frames, joint_count, 3), dtype=np.float64)
    rotations[:, 0] = local_rotations[:, 0]
    positions[:, 0] = rest_joints[0]

    for joint in range(1, joint_count):
        parent = int(parents[joint])
        if parent < 0 or parent >= joint:
            raise ValueError(f"Invalid parent {parent} for joint {joint}")
        rotations[:, joint] = rotations[:, parent] @ local_rotations[:, joint]
        rest_offset = rest_joints[joint] - rest_joints[parent]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "fij,j->fi", rotations[:, parent], rest_offset
        )

    positions += translations[:, None, :]
    return positions, rotations


def prepare_motion(
    result: np.lib.npyio.NpzFile,
    model_data: np.lib.npyio.NpzFile,
    keep_world_position: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices, rest_joints = shaped_template_and_joints(result, model_data)
    parents = np.asarray(model_data["kintree_table"][0], dtype=np.int64).copy()
    parents[0] = -1
    translations = np.asarray(result["trans"], dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(f"trans must have shape (frames, 3), got {translations.shape}")

    positions, rotations = global_joint_motion(
        full_pose(result), rest_joints, parents, translations
    )

    vertices_mj = vertices @ SMPLX_TO_MUJOCO.T
    rest_joints_mj = rest_joints @ SMPLX_TO_MUJOCO.T
    positions_mj = positions @ SMPLX_TO_MUJOCO.T
    rotations_mj = np.einsum(
        "ab,fjbc,cd->fjad",
        SMPLX_TO_MUJOCO,
        rotations,
        SMPLX_TO_MUJOCO.T,
    )

    if not keep_world_position:
        horizontal_offset = np.array(
            [-positions_mj[0, 0, 0], -positions_mj[0, 0, 1], 0.0]
        )
        positions_mj += horizontal_offset[None, None, :]

    quaternions_mj = matrices_to_quaternions(rotations_mj)
    return vertices_mj, rest_joints_mj, positions_mj, quaternions_mj
