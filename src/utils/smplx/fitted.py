"""Load fitted SMPL-X motion on an aligned recording's source-frame timeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.smplx.kinematics import (
    matrices_to_quaternions,
    prepare_motion,
    require_keys,
    shaped_template_and_joints,
)
from utils.smplx.models import (
    REQUIRED_MODEL_KEYS,
    REQUIRED_RESULT_KEYS,
    SMPLX_TO_MUJOCO,
)
from utils.smplx.neutral import (
    NeutralAvatar,
    estimate_glasses_bridge,
)


@dataclass(frozen=True)
class FittedMotion:
    """A shaped avatar and fitted joint motion sampled on the paired timeline."""

    avatar: NeutralAvatar
    positions: np.ndarray
    rotations: np.ndarray
    quaternions: np.ndarray
    result_path: Path
    source_frame_numbers: np.ndarray
    world_rotation: np.ndarray
    world_translation: np.ndarray
    registration_median_error_m: float
    registration_p95_error_m: float


def match_source_frames(
    fitted_frame_numbers: np.ndarray, requested_frame_numbers: np.ndarray
) -> np.ndarray:
    """Return exact fitted-row indices for source frames without interpolation."""
    fitted = np.asarray(fitted_frame_numbers, dtype=np.int64)
    requested = np.asarray(requested_frame_numbers, dtype=np.int64)
    if fitted.ndim != 1 or requested.ndim != 1:
        raise ValueError("Source frame numbers must be one-dimensional")
    if len(np.unique(fitted)) != len(fitted):
        raise ValueError("Fitted SMPL-X source frame numbers contain duplicates")
    lookup = {int(frame): index for index, frame in enumerate(fitted)}
    missing = sorted({int(frame) for frame in requested if int(frame) not in lookup})
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            "Fitted SMPL-X result does not cover aligned Motive source frames: "
            f"{preview}{suffix} ({len(missing)} missing)"
        )
    return np.asarray([lookup[int(frame)] for frame in requested], dtype=np.int64)


def register_fitted_world(
    fitted_markers: np.ndarray,
    fitted_labels: np.ndarray,
    reference_markers: np.ndarray,
    reference_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Register fitted marker coordinates to a paired recording's world frame.

    MoSh++ consumes the C3D coordinates, while the Motive CSV used by the paired
    session can use a different fixed export convention.  Estimate that one
    rigid transform from same-frame, same-label observations instead of hiding
    a capture-specific axis flip in the viewer.
    """
    fitted = np.asarray(fitted_markers, dtype=np.float64)
    reference = np.asarray(reference_markers, dtype=np.float64)
    labels = [str(label) for label in np.asarray(fitted_labels).reshape(-1)]
    if fitted.ndim != 3 or fitted.shape[2] != 3:
        raise ValueError("Fitted markers must have shape (frames, markers, 3)")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference markers must have shape (frames, markers, 3)")
    if len(fitted) != len(reference):
        raise ValueError("Fitted and reference marker timelines differ")
    if fitted.shape[1] != len(labels):
        raise ValueError("Fitted marker labels and columns differ")

    fitted_by_name = {name.rsplit(":", 1)[-1]: i for i, name in enumerate(labels)}
    reference_by_name = {
        str(name).rsplit(":", 1)[-1]: i for i, name in enumerate(reference_labels)
    }
    common = sorted(set(fitted_by_name).intersection(reference_by_name))
    if len(common) < 3:
        raise ValueError("Fewer than three shared markers for world registration")

    source_columns = [fitted_by_name[name] for name in common]
    target_columns = [reference_by_name[name] for name in common]
    source = fitted[:, source_columns].reshape(-1, 3)
    target = reference[:, target_columns].reshape(-1, 3)
    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & (np.linalg.norm(source, axis=1) > 1e-8)
    )
    source, target = source[valid], target[valid]
    if len(source) < 30:
        raise ValueError("Too few valid marker pairs for world registration")

    def solve(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_mean, b_mean = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - a_mean).T @ (b - b_mean))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = b_mean - a_mean @ rotation.T
        return rotation, translation

    rotation, translation = solve(source, target)
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    cutoff = max(0.005, 5.0 * float(np.median(residual)))
    inliers = residual <= cutoff
    if int(inliers.sum()) >= 30:
        rotation, translation = solve(source[inliers], target[inliers])
        residual = np.linalg.norm(
            source @ rotation.T + translation - target, axis=1
        )
    median = float(np.median(residual))
    p95 = float(np.percentile(residual, 95))
    if median > 0.005 or p95 > 0.02:
        raise ValueError(
            "MoSh++/paired-session world registration is inconsistent: "
            f"median={median*1000:.2f} mm, p95={p95*1000:.2f} mm"
        )
    return rotation, translation, median, p95


def load_fitted_motion(
    result_path: Path,
    model_path: Path,
    requested_frame_numbers: np.ndarray,
    reference_markers: np.ndarray,
    reference_marker_names: list[str],
) -> FittedMotion:
    """Reconstruct fitted motion and select exact aligned Motive source frames."""
    result_path, model_path = Path(result_path), Path(model_path)
    with np.load(result_path, allow_pickle=False) as result, np.load(
        model_path, allow_pickle=False
    ) as model:
        require_keys(result, REQUIRED_RESULT_KEYS, "SMPL-X result")
        require_keys(model, REQUIRED_MODEL_KEYS, "SMPL-X model")
        if "source_frame_numbers" not in result.files:
            raise KeyError(
                "SMPL-X result is missing source_frame_numbers; refusing to infer "
                "its relationship to the paired recording"
            )
        fitted_frames = np.asarray(result["source_frame_numbers"], dtype=np.int64)
        vertices, joints, positions, quaternions = prepare_motion(
            result, model, keep_world_position=True
        )
        if len(fitted_frames) != len(positions):
            raise ValueError(
                "SMPL-X source-frame mapping and fitted motion lengths differ"
            )
        indices = match_source_frames(fitted_frames, requested_frame_numbers)
        if "markers" not in result.files or "labels" not in result.files:
            raise KeyError(
                "SMPL-X result is missing markers/labels required to register "
                "its C3D world frame to the paired Motive CSV"
            )
        fitted_markers = np.asarray(result["markers"], dtype=np.float64)
        if fitted_markers.shape[:2] != (len(fitted_frames), len(result["labels"])):
            raise ValueError("Fitted marker observations and labels differ")
        fitted_markers = fitted_markers[indices] @ SMPLX_TO_MUJOCO.T
        world_rotation, world_translation, median_error, p95_error = (
            register_fitted_world(
                fitted_markers,
                result["labels"],
                reference_markers,
                reference_marker_names,
            )
        )
        native_vertices, _ = shaped_template_and_joints(result, model)
        faces = np.asarray(model["f"], dtype=np.int64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        parents = np.asarray(model["kintree_table"][0], dtype=np.int64).copy()
        parents[0] = -1

    rotations = Rotation.from_quat(
        quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]
    ).as_matrix().reshape(*quaternions.shape[:2], 3, 3)
    positions = positions @ world_rotation.T + world_translation
    rotations = np.einsum("ab,fjbc->fjac", world_rotation, rotations)
    quaternions = matrices_to_quaternions(rotations)
    bridge = estimate_glasses_bridge(native_vertices, faces) @ SMPLX_TO_MUJOCO.T
    avatar = NeutralAvatar(vertices, faces, weights, joints, parents, bridge)
    return FittedMotion(
        avatar=avatar,
        positions=positions[indices],
        rotations=rotations[indices],
        quaternions=quaternions[indices],
        result_path=result_path,
        source_frame_numbers=fitted_frames[indices],
        world_rotation=world_rotation,
        world_translation=world_translation,
        registration_median_error_m=median_error,
        registration_p95_error_m=p95_error,
    )
