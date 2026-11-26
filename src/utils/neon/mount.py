"""Attach the shipped Neon CAD and optical debug frame to a tracked head."""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.neon.model import values
from utils.paths import ASSETS_DIR
from utils.smplx.models import SMPLX_TO_MUJOCO

# Approximate underside of the bridge in the reference CAD, metres.
CAD_BRIDGE_M = np.array([0, 0.0452, 0.0136])


def mount_transform(
    bridge_from_head: np.ndarray, settings: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Pose in head bind axes (MuJoCo Z-up); angles rotate the CAD locally."""
    rotation = (
        SMPLX_TO_MUJOCO
        @ Rotation.from_euler(
            "xyz", settings.get("rotation_deg_xyz", [0, 0, 0]), degrees=True
        ).as_matrix()
    )
    offset = np.asarray(settings.get("translation_m", [0, 0, 0]), dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Glasses translation_m must contain three finite values")
    position = np.asarray(bridge_from_head) + offset - rotation @ CAD_BRIDGE_M
    return position, Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


def attach_neon(
    root: ET.Element, head: ET.Element, bridge_from_head: np.ndarray, settings: dict
) -> None:
    """Merge real CAD meshes, intrinsics and camera axes into a head body."""
    source_dir = ASSETS_DIR / "neon"
    source = ET.parse(source_dir / "neon.xml").getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for mesh in source.find("asset"):
        element = copy.deepcopy(mesh)
        element.set("file", str((source_dir / element.attrib["file"]).resolve()))
        asset.append(element)
    root.append(copy.deepcopy(source.find("custom")))
    glasses = copy.deepcopy(source.find("worldbody/body"))
    position, quaternion = mount_transform(bridge_from_head, settings)
    glasses.set("pos", values(position))
    glasses.set("quat", values(quaternion))
    head.append(glasses)
    ET.SubElement(
        head,
        "site",
        name="nose_bridge_reference",
        pos=values(bridge_from_head),
        size="0.002",
        rgba="1 0.75 0.1 1",
        group="5",
    )


def load_mount(path: Path | None) -> dict:
    if path is None:
        return {"translation_m": [0.0, 0.0, 0.0], "rotation_deg_xyz": [0.0, 0.0, 0.0]}
    return json.loads(Path(path).read_text())
