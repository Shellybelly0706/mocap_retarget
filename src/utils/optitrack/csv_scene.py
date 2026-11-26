"""Reusable optitrack csv scene helpers."""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

import mujoco
import numpy as np

from utils.optitrack.csv import (
    CsvExport,
)
from utils.optitrack.skeleton import SKELETON_PARENTS

HIDDEN_POSITION = np.array([0.0, 0.0, -100.0])


CATEGORY_STYLES = {
    "Bone": (0.014, "0.1 0.65 1 1"),
    "Skeleton": (0.009, "0.08 0.48 1 0.9"),
    "Bone Marker": (0.012, "0.15 1 0.35 0.9"),
    "Rigid Body Marker": (0.018, "1 0.45 0.05 1"),
    "Marker": (0.006, "0.75 0.8 0.85 0.38"),
}


CATEGORY_KEYS = {
    49: "Bone",
    50: "Bone Marker",
    51: "Rigid Body",
    52: "Rigid Body Marker",
    53: "Marker",
    54: "Skeleton",
}


@dataclass
class Entity:
    name: str
    category: str
    positions: np.ndarray
    quaternions: np.ndarray
    geom_xml: str


def identity_quaternions(frame_count: int, object_count: int) -> np.ndarray:
    result = np.zeros((frame_count, object_count, 4))
    result[:, :, 0] = 1.0
    return result


def z_axis_quaternions(vectors: np.ndarray) -> np.ndarray:
    """Return MuJoCo WXYZ quaternions rotating local +Z onto each vector."""
    result = np.full((len(vectors), 4), np.nan)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (lengths > 1e-9)
    directions = vectors[valid] / lengths[valid, None]
    regular = directions[:, 2] > -1.0 + 1e-8
    quaternions = np.empty((len(directions), 4))
    scale = np.sqrt(2.0 * (1.0 + directions[regular, 2]))
    quaternions[regular, 0] = scale / 2.0
    quaternions[regular, 1] = -directions[regular, 1] / scale
    quaternions[regular, 2] = directions[regular, 0] / scale
    quaternions[regular, 3] = 0.0
    # A 180-degree rotation around X maps +Z to -Z.
    quaternions[~regular] = np.array([0.0, 1.0, 0.0, 0.0])
    result[valid] = quaternions
    return result


def make_skeleton_entities(csv_export: CsvExport) -> list[Entity]:
    """Create fixed-length capsules joining Motive skeleton parent/child bones."""
    entities: list[Entity] = []
    indices = {name: index for index, name in enumerate(csv_export.bone_names)}
    root_by_namespace: dict[str, str] = {}
    for name in csv_export.bone_names:
        namespace, separator, local_name = name.rpartition(":")
        if separator and local_name == namespace:
            root_by_namespace[namespace] = name

    radius, rgba = CATEGORY_STYLES["Skeleton"]
    for child_name, child_index in indices.items():
        namespace, separator, local_name = child_name.rpartition(":")
        if not separator:
            continue
        parent_local = SKELETON_PARENTS.get(local_name)
        if local_name in ("Ab", "LThigh", "RThigh"):
            parent_name = root_by_namespace.get(namespace)
        elif parent_local is not None:
            parent_name = f"{namespace}:{parent_local}"
        else:
            continue
        # A hand-only Motive skeleton can export its wrist as the namespace
        # root without a separate LHand/RHand bone.
        if (
            parent_name not in indices
            and parent_local in ("LHand", "RHand")
            and local_name.endswith("1")
        ):
            parent_name = root_by_namespace.get(namespace)
        if parent_name not in indices:
            continue

        parent_positions = csv_export.bone_positions[:, indices[parent_name]]
        child_positions = csv_export.bone_positions[:, child_index]
        vectors = child_positions - parent_positions
        lengths = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & (lengths > 1e-9)
        if not valid.any():
            continue
        length = float(np.median(lengths[valid]))
        entities.append(
            Entity(
                name=f"{parent_name} -> {child_name}",
                category="Skeleton",
                positions=(parent_positions + child_positions) / 2.0,
                quaternions=z_axis_quaternions(vectors),
                geom_xml=(
                    f'<geom type="capsule" size="{radius:.5g} {length / 2:.7g}" '
                    f'rgba="{rgba}" contype="0" conaffinity="0"/>'
                ),
            )
        )
    return entities


def make_entities(csv_export: CsvExport) -> list[Entity]:
    """Create skeleton links, point tracks, and rigid-body entities."""
    entities: list[Entity] = make_skeleton_entities(csv_export)

    def add_points(
        category: str,
        names: list[str],
        positions: np.ndarray,
    ) -> None:
        radius, rgba = CATEGORY_STYLES[category]
        quaternions = identity_quaternions(len(csv_export.times), len(names))
        for index, name in enumerate(names):
            entities.append(
                Entity(
                    name=name,
                    category=category,
                    positions=positions[:, index],
                    quaternions=quaternions[:, index],
                    geom_xml=(
                        f'<geom type="sphere" size="{radius:.5g}" '
                        f'rgba="{rgba}" contype="0" conaffinity="0"/>'
                    ),
                )
            )

    add_points("Bone", csv_export.bone_names, csv_export.bone_positions)
    for category, (names, positions) in csv_export.point_groups.items():
        add_points(category, names, positions)

    for index, name in enumerate(csv_export.rigid_names):
        entities.append(
            Entity(
                name=name,
                category="Rigid Body",
                positions=csv_export.rigid_positions[:, index],
                quaternions=csv_export.rigid_quaternions[:, index],
                geom_xml=(
                    '<geom type="box" size="0.085 0.055 0.035" '
                    'rgba="1 0.15 0.1 0.85" contype="0" conaffinity="0"/>'
                ),
            )
        )
    return entities


def build_model(entities: list[Entity], timestep: float) -> mujoco.MjModel:
    bodies = []
    for index, entity in enumerate(entities):
        body_name = escape(f"entity_{index:03d}_{entity.category}_{entity.name}")
        bodies.append(
            f"""<body name="{body_name}" mocap="true">
                   {entity.geom_xml}
                </body>"""
        )
    xml = f"""
    <mujoco model="1105_csv_take">
      <option timestep="{timestep:.12g}" gravity="0 0 -9.81"/>
      <visual>
        <headlight ambient="0.48 0.48 0.48" diffuse="0.75 0.75 0.75"/>
        <rgba haze="0.13 0.16 0.2 1"/>
      </visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.12 0.12 0.14" rgb2="0.22 0.22 0.25"/>
        <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.08"/>
      </asset>
      <worldbody>
        <light pos="0 -1 7" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def update_scene(
    data: mujoco.MjData,
    entities: list[Entity],
    frame: int,
    visible: dict[str, bool],
) -> None:
    for index, entity in enumerate(entities):
        position = entity.positions[frame]
        quaternion = entity.quaternions[frame]
        if (
            not visible[entity.category]
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
        ):
            data.mocap_pos[index] = HIDDEN_POSITION
            data.mocap_quat[index] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            data.mocap_pos[index] = position
            data.mocap_quat[index] = quaternion
