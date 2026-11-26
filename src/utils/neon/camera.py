"""Factory scene-camera intrinsics and OpenCV-to-MuJoCo conventions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SceneCameraCalibration:
    serial: str
    matrix: np.ndarray
    distortion: np.ndarray
    resolution: tuple[int, int] = (1600, 1200)

    def mjcf_attributes(self) -> dict[str, str]:
        """Use pixel focal lengths and offsets from the image centre.

        MuJoCo uses an OpenGL camera (+X right, +Y up, -Z forward).
        Its principal-point offsets have the opposite sign to OpenCV's
        displacement from the image centre, for both image coordinates.
        MuJoCo's renderer is pinhole-only; distortion is stored separately.
        """
        width, height = self.resolution
        fx, fy = self.matrix[0, 0], self.matrix[1, 1]
        cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        return {
            "resolution": f"{width} {height}",
            "sensorsize": "1 1",
            "focalpixel": f"{fx:.12g} {fy:.12g}",
            "principalpixel": f"{width / 2 - cx:.12g} {height / 2 - cy:.12g}",
        }


def load_scene_camera_calibration(path: Path) -> SceneCameraCalibration:
    """Read a Pupil Labs scene_camera.json file or a successful API response."""
    payload = json.loads(Path(path).read_text())
    if "result" in payload:
        if payload.get("code") != 200:
            raise ValueError("Pupil Labs calibration response was not successful")
        payload = payload["result"]
    matrix = np.asarray(payload["camera_matrix"], dtype=float)
    distortion = np.asarray(payload["dist_coefs"], dtype=float).reshape(-1)
    resolution = tuple(int(x) for x in payload.get("resolution", [1600, 1200]))
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Scene camera matrix must be a finite 3x3 array")
    if not np.allclose(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0:
        raise ValueError(
            "MuJoCo requires zero skew and a standard pinhole camera matrix"
        )
    if (
        min(matrix[0, 0], matrix[1, 1]) <= 0
        or len(resolution) != 2
        or min(resolution) <= 0
    ):
        raise ValueError("Focal lengths and image dimensions must be positive")
    if distortion.size != 8 or not np.isfinite(distortion).all():
        raise ValueError(
            "Expected eight finite OpenCV rational distortion coefficients"
        )
    return SceneCameraCalibration(
        str(payload["serial_number"]), matrix, distortion, resolution
    )
