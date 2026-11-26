"""Reusable optitrack c3d scene helpers."""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from utils.optitrack.c3d import (
    C3DMotion,
)
from utils.optitrack.skeleton import (
    SkeletonMotion,
)

HIDDEN_POSITION = np.array([0.0, 0.0, -100.0])


MARKER_RGBA = np.array([0.1, 0.9, 0.3, 1.0], dtype=np.float32)


JOINT_RGBA = np.array([1.0, 0.72, 0.08, 1.0], dtype=np.float32)


BONE_RGBA = np.array([0.08, 0.48, 1.0, 0.95], dtype=np.float32)


def build_model(timestep: float, floor_height: float) -> mujoco.MjModel:
    xml = f"""
    <mujoco model="labeled_c3d_markers">
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
        <geom name="floor" type="plane" pos="0 0 {floor_height:.8g}"
              size="5 5 0.1" material="grid"/>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def initialize_marker_geoms(scene, labels: Sequence[str], marker_size: float) -> None:
    if len(labels) > scene.maxgeom:
        raise ValueError(
            f"Viewer scene supports {scene.maxgeom} user geoms, but C3D has {len(labels)}"
        )
    scene.ngeom = len(labels)
    size = np.array([marker_size, 0.0, 0.0])
    matrix = np.eye(3).ravel()
    for index, label in enumerate(labels):
        geom = scene.geoms[index]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            HIDDEN_POSITION,
            matrix,
            MARKER_RGBA,
        )
        geom.label = label


def initialize_skeleton_geoms(
    scene,
    marker_count: int,
    skeleton: SkeletonMotion,
    joint_size: float,
    bone_width: float,
) -> None:
    required_geoms = marker_count + len(skeleton.node_names) + len(skeleton.edges)
    if required_geoms > scene.maxgeom:
        raise ValueError(
            f"Viewer scene supports {scene.maxgeom} user geoms, but "
            f"markers plus skeleton require {required_geoms}"
        )
    scene.ngeom = required_geoms
    matrix = np.eye(3).ravel()
    joint_size_array = np.array([joint_size, 0.0, 0.0])
    for node_index in range(len(skeleton.node_names)):
        geom = scene.geoms[marker_count + node_index]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            joint_size_array,
            HIDDEN_POSITION,
            matrix,
            JOINT_RGBA,
        )
        geom.label = ""
    edge_start = marker_count + len(skeleton.node_names)
    for edge_index in range(len(skeleton.edges)):
        geom = scene.geoms[edge_start + edge_index]
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            bone_width,
            HIDDEN_POSITION,
            HIDDEN_POSITION + np.array([0.0, 0.0, 1e-6]),
        )
        geom.rgba[:] = BONE_RGBA
        geom.label = ""


def update_marker_geoms(
    scene,
    motion: C3DMotion,
    labels: Sequence[str],
    frame: int,
    show_labels: bool,
    show_markers: bool,
) -> None:
    for index, label in enumerate(labels):
        geom = scene.geoms[index]
        if motion.valid[frame, index]:
            geom.pos[:] = motion.positions[frame, index]
            geom.rgba[3] = 1.0 if show_markers else 0.0
            geom.label = label if show_labels else ""
        else:
            geom.pos[:] = HIDDEN_POSITION
            geom.rgba[3] = 0.0
            geom.label = ""


def update_skeleton_geoms(
    scene,
    marker_count: int,
    skeleton: SkeletonMotion,
    frame: int,
    show_skeleton: bool,
    bone_width: float,
) -> None:
    for node_index in range(len(skeleton.node_names)):
        geom = scene.geoms[marker_count + node_index]
        if show_skeleton and skeleton.valid[frame, node_index]:
            geom.pos[:] = skeleton.positions[frame, node_index]
            geom.rgba[:] = JOINT_RGBA
        else:
            geom.pos[:] = HIDDEN_POSITION
            geom.rgba[3] = 0.0
    edge_start = marker_count + len(skeleton.node_names)
    for edge_index, (parent, child) in enumerate(skeleton.edges):
        geom = scene.geoms[edge_start + edge_index]
        if (
            show_skeleton
            and skeleton.valid[frame, parent]
            and skeleton.valid[frame, child]
        ):
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                bone_width,
                skeleton.positions[frame, parent],
                skeleton.positions[frame, child],
            )
            geom.rgba[:] = BONE_RGBA
        else:
            geom.pos[:] = HIDDEN_POSITION
            geom.rgba[3] = 0.0
