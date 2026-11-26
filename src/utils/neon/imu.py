"""Neon IMU mounting conventions, native orientations and world registration.

R_A_B maps vectors from frame B into A. IMU quaternions map the local IMU
frame into the magnetic/gravity world, whose heading differs from OptiTrack.
No timestamp correction or position integration is performed here.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from utils.paths import ASSETS_DIR


def imu_reference() -> dict:
    return json.loads((ASSETS_DIR / "neon" / "imu_reference.json").read_text())


def camera_to_imu() -> tuple[np.ndarray, np.ndarray]:
    """Return manufacturer's nominal scene-camera -> IMU point transform, metres."""
    reference = imu_reference()
    return (
        Rotation.from_euler(
            "x", reference["rotation_imu_from_camera_x_degrees"], degrees=True
        ).as_matrix(),
        np.asarray(reference["translation_imu_from_camera_m"], dtype=float),
    )


def imu_pose_in_cad(
    camera_position: np.ndarray, camera_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Place the top-bar IMU relative to the existing CAD optical reference."""
    rotation_i_c, translation_i_c = camera_to_imu()
    rotation_a_i = np.asarray(camera_rotation) @ rotation_i_c.T
    position_a_i = np.asarray(camera_position) - rotation_a_i @ translation_i_c
    return position_a_i, rotation_a_i


@dataclass
class SampledImu:
    rotations: np.ndarray
    source_left: np.ndarray
    source_right: np.ndarray
    fractions: np.ndarray
    gaps_s: np.ndarray
    report: dict


def sample_native_imu(
    recording: Path, timestamps_ns: np.ndarray, max_gap_s: float = 0.05
) -> SampledImu:
    """SLERP native module quaternions onto video timestamps without extrapolation."""
    recording = Path(recording)
    dtype_path = recording / "imu.dtype"
    raw_path = recording / "imu ps1.raw"
    dtype = np.dtype(ast.literal_eval(dtype_path.read_text())).newbyteorder("<")
    required = {"timestamp_ns", *(f"quaternion_{axis}" for axis in "wxyz")}
    if (
        dtype.names is None
        or not required.issubset(dtype.names)
        or raw_path.stat().st_size % dtype.itemsize
    ):
        raise ValueError("Native module IMU format is incomplete or inconsistent")
    raw = np.fromfile(raw_path, dtype=dtype)
    raw_times = np.asarray(raw["timestamp_ns"], dtype=np.int64)
    if len(raw_times) < 2 or np.any(np.diff(raw_times) <= 0):
        raise ValueError("IMU timestamps must be strictly increasing")
    quats = np.column_stack([raw[f"quaternion_{axis}"] for axis in "wxyz"]).astype(
        float
    )
    norms = np.linalg.norm(quats, axis=1)
    valid = np.isfinite(quats).all(axis=1) & (norms > 0.9) & (norms < 1.1)
    indices = np.flatnonzero(valid)
    times = raw_times[valid]
    targets = np.asarray(timestamps_ns, dtype=np.int64)
    if len(times) < 2 or len(targets) < 2 or np.any(np.diff(targets) <= 0):
        raise ValueError(
            "Need at least two valid IMU samples and increasing target times"
        )
    if targets[0] < times[0] or targets[-1] > times[-1]:
        raise ValueError("Native IMU does not cover the aligned video interval")
    right = np.searchsorted(times, targets, side="left").clip(1, len(times) - 1)
    left = right - 1
    spans = times[right] - times[left]
    fractions = (targets - times[left]) / spans
    gaps = spans / 1e9
    if np.any(gaps > max_gap_s):
        raise ValueError(
            f"IMU interpolation would bridge a gap longer than {max_gap_s}s"
        )
    epoch = int(times[0])
    rotations = Slerp(
        (times - epoch) / 1e9, Rotation.from_quat(quats[valid], scalar_first=True)
    )((targets - epoch) / 1e9).as_matrix()
    report = {
        "source": "native Neon module imu ps1.raw (not extimu/phone)",
        "sources": {
            key: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in (("raw", raw_path), ("dtype", dtype_path))
        },
        "sample_count": len(raw),
        "invalid_quaternion_count": int((~valid).sum()),
        "observed_median_rate_hz": float(1e9 / np.median(np.diff(raw_times))),
        "quaternion_order": "WXYZ; local IMU to magnetic/gravity world",
        "sampling": "SLERP on unchanged canonical Neon video timestamps; no extrapolation",
        "clock_correction_ns": 0,
        "maximum_interpolation_gap_s": float(gaps.max()),
    }
    return SampledImu(rotations, indices[left], indices[right], fractions, gaps, report)


def register_imu_world(
    head_rotations: np.ndarray,
    imu_rotations: np.ndarray,
    times_s: np.ndarray,
    fit_start_s: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Fit unknown world yaw and constant head<-IMU rotation from paired motion.

    Both worlds are assumed Z-up. Fit a constant mount and world yaw jointly:
    R_mocap_head(t) R_head_imu = Rz(yaw) R_imuWorld_imu(t).
    Require a usable held-out interval and a well-conditioned motion fit.
    """
    head, imu = np.asarray(head_rotations), np.asarray(imu_rotations)
    times = np.asarray(times_s)
    if head.shape != imu.shape or head.shape != (len(times), 3, 3):
        raise ValueError(
            "Head and IMU rotations must have matching (frames,3,3) shapes"
        )
    selected = times >= fit_start_s
    if selected.sum() < 60:
        raise ValueError(
            "IMU registration needs at least 60 paired frames after the fit start"
        )
    split = fit_start_s + (times[-1] - fit_start_s) * 0.6
    train = selected & (times <= split)
    test = selected & ~train
    if train.sum() < 30 or test.sum() < 20:
        raise ValueError("Insufficient training/held-out motion for IMU registration")

    def solve(mask):
        a, b = head[mask], imu[mask]

        def residual(parameters):
            x = Rotation.from_rotvec(parameters[:3]).as_matrix()
            y = Rotation.from_euler("z", parameters[3]).as_matrix()
            return (
                Rotation.from_matrix((a @ x).transpose(0, 2, 1) @ (y @ b))
                .as_rotvec()
                .ravel()
            )

        initial = np.r_[Rotation.from_matrix(a[0].T @ b[0]).as_rotvec(), 0.0]
        solution = least_squares(
            residual, initial, loss="soft_l1", f_scale=np.deg2rad(2), max_nfev=300
        )
        if not solution.success:
            raise ValueError("IMU/head orientation registration did not converge")
        singular = np.linalg.svd(solution.jac, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        if condition > 100:
            raise ValueError(
                f"IMU/head mount is not observable from this motion (condition {condition:.1f})"
            )
        return solution.x, condition

    def errors(parameters, mask):
        x = Rotation.from_rotvec(parameters[:3]).as_matrix()
        y = Rotation.from_euler("z", parameters[3]).as_matrix()
        return np.rad2deg(
            Rotation.from_matrix(
                (head[mask] @ x).transpose(0, 2, 1) @ (y @ imu[mask])
            ).magnitude()
        )

    training, training_condition = solve(train)
    heldout = errors(training, test)
    if np.quantile(heldout, 0.9) > 5:
        raise ValueError(
            "IMU/head orientation fit failed held-out validation (90th percentile >5 degrees)"
        )
    parameters, condition = solve(selected)
    residual = errors(parameters, selected)
    if np.quantile(residual, 0.9) > 5:
        raise ValueError(
            "IMU/head registration residual exceeds 5 degrees at the 90th percentile"
        )
    world = Rotation.from_euler("z", parameters[3]).as_matrix()
    mount = Rotation.from_rotvec(parameters[:3]).as_matrix()
    all_residual = np.rad2deg(
        Rotation.from_matrix(
            (head @ mount).transpose(0, 2, 1) @ (world @ imu)
        ).magnitude()
    )
    report = {
        "method": "robust constant head<-IMU rotation and world yaw; both worlds assumed Z-up",
        "rotation_mocap_from_imu_world": world.tolist(),
        "rotation_head_from_imu": mount.tolist(),
        "world_yaw_degrees": float(np.rad2deg(parameters[3])),
        "fit_start_s": fit_start_s,
        "fit_start_reason": "omit solved-head startup transient from parameter estimation only; all playback frames retained",
        "fit_frames": int(selected.sum()),
        "condition_number": condition,
        "median_residual_degrees": float(np.median(residual)),
        "p90_residual_degrees": float(np.quantile(residual, 0.9)),
        "maximum_residual_degrees": float(residual.max()),
        "heldout": {
            "training_end_s": split,
            "training_frames": int(train.sum()),
            "test_frames": int(test.sum()),
            "training_condition_number": training_condition,
            "p90_residual_degrees": float(np.quantile(heldout, 0.9)),
        },
        "per_frame_residual_degrees": all_residual.tolist(),
        "limitations": "Estimated orientation registration, not surveyed extrinsics; no position from IMU; clock agreement still assumed; magnetic disturbance/drift may affect heading.",
    }
    return world @ imu, report


def save_imu_report(
    directory: Path, sampled: SampledImu, registration: dict, timestamps_ns: np.ndarray
) -> None:
    """Store separate derived IMU provenance; leave existing frame alignment intact."""
    directory = Path(directory)
    report = {
        "imu": sampled.report,
        "registration": registration,
        "installation": imu_reference(),
        "paired_session": {
            name: {
                "path": str((directory / name).resolve()),
                "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            }
            for name in ("alignment.json", "frame_alignment.csv", "mocap_aligned.csv")
        },
    }
    (directory / "imu_attachment.json").write_text(json.dumps(report, indent=2) + "\n")
    quats = Rotation.from_matrix(sampled.rotations).as_quat(scalar_first=True)
    with (directory / "imu_attachment.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "aligned_frame",
                "timestamp_ns",
                "imu_source_left",
                "imu_source_right",
                "slerp_fraction",
                "interpolation_span_s",
                "quaternion_w",
                "quaternion_x",
                "quaternion_y",
                "quaternion_z",
                "head_imu_residual_degrees",
            ]
        )
        for index, stamp in enumerate(timestamps_ns):
            writer.writerow(
                [
                    index,
                    int(stamp),
                    int(sampled.source_left[index]),
                    int(sampled.source_right[index]),
                    sampled.fractions[index],
                    sampled.gaps_s[index],
                    *quats[index],
                    registration["per_frame_residual_degrees"][index],
                ]
            )
