"""Visual opening of the CAD temple arms without changing the sensor module.

The shipped CAD has separate rigid arms. Opening them about nominal hinges is
a visual accommodation to a neutral avatar, not a measured participant fit.
Original STL assets, the front frame, camera and IMU geometry remain unchanged.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from utils.neon.model import values
from utils.paths import ASSETS_DIR


@lru_cache(maxsize=1)
def frame_parts():
    """Recover the front and two arms from the two STL triangle chunks."""
    mesh = trimesh.util.concatenate(
        [trimesh.load(ASSETS_DIR / "neon/meshes" / f"frame_{i}.stl") for i in (0, 1)]
    )
    mesh.merge_vertices()
    parts = {}
    front = []
    for part in mesh.split(only_watertight=False):
        if part.bounds[0, 2] < -0.1:
            # CAD +X points to the wearer's left.
            side = "left" if part.vertices[:, 0].mean() > 0 else "right"
            if side in parts:
                raise ValueError("Neon CAD no longer has two separate temple arms")
            # Nominal vertical hinge through the front end of each arm.
            end = part.vertices[:, 2] > part.bounds[1, 2] - 0.002
            parts[side] = (part, part.vertices[end].mean(axis=0))
        else:
            front.append(part)
    if set(parts) != {"left", "right"}:
        raise ValueError("Cannot identify the Neon CAD temple arms")
    return trimesh.util.concatenate(front), parts


def hinge_quaternion(side: str, opening_deg: float) -> np.ndarray:
    sign = -1 if side == "left" else 1
    return Rotation.from_euler("y", sign * opening_deg, degrees=True).as_quat()[
        [3, 0, 1, 2]
    ]


class TempleFitter:
    """Choose a static opening using approximate signed head-surface clearance."""

    def __init__(self, head_vertices: np.ndarray, faces: np.ndarray):
        head = trimesh.Trimesh(head_vertices, faces, process=False)
        self.vertices = head.vertices
        self.normals = head.vertex_normals
        self.tree = cKDTree(head.vertices)

    def fit(self, position: np.ndarray, rotation: np.ndarray) -> dict:
        report = {"purpose": "visual neutral-head fit; nominal hinges, not calibrated"}
        _, parts = frame_parts()
        for side, (arm, pivot) in parts.items():
            # Sample the actual arm, including its ear hook. Nearest-vertex
            # normals approximate clearance; multi-view rendering is also needed.
            points = arm.vertices[::8] - pivot
            sign = -1 if side == "left" else 1

            def clearance(angle):
                opened = Rotation.from_euler("y", sign * angle, degrees=True).apply(
                    points
                )
                candidate = (opened + pivot) @ rotation.T + position
                _, nearest = self.tree.query(candidate)
                signed = np.einsum(
                    "ij,ij->i",
                    candidate - self.vertices[nearest],
                    self.normals[nearest],
                )
                return float(np.min(signed))

            before = clearance(0)
            angle = 0.0
            after = before
            # Keep the smallest opening with 1 mm approximate skin clearance.
            while after < 0.001 and angle < 15:
                angle += 0.25
                after = clearance(angle)
            report[side] = {
                "opening_deg": angle,
                "min_clearance_before_m": before,
                "min_clearance_after_m": after,
                "clearance_reached": after >= 0.001,
            }
        return report


def attach_temple_hinges(root: ET.Element, fit: dict) -> None:
    """Replace the frame chunks with the same triangles in three rigid bodies."""
    asset = root.find("asset")
    glasses = root.find(".//body[@name='neon_glasses']")
    template = copy.deepcopy(glasses.find("geom[@name='neon_frame_0_visual']"))
    for index in (0, 1):
        asset.remove(asset.find(f"mesh[@name='neon_frame_{index}']"))
        glasses.remove(glasses.find(f"geom[@name='neon_frame_{index}_visual']"))
    front, parts = frame_parts()
    for name, part, pivot in [
        ("front", front, np.zeros(3)),
        *[(side, part, pivot) for side, (part, pivot) in parts.items()],
    ]:
        mesh_name = f"neon_frame_{name}"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            vertex=values(part.vertices - pivot),
            face=" ".join(map(str, part.faces.reshape(-1))),
            maxhullvert="64",
        )
        parent = glasses
        if name != "front":
            parent = ET.SubElement(
                glasses,
                "body",
                name=f"neon_temple_{name}",
                pos=values(pivot),
                quat=values(hinge_quaternion(name, fit[name]["opening_deg"])),
            )
        geom = copy.deepcopy(template)
        geom.set("name", f"{mesh_name}_visual")
        geom.set("mesh", mesh_name)
        parent.append(geom)
