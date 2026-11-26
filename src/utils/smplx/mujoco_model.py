"""Reusable smplx mujoco model helpers."""

from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np

from utils.smplx.models import (
    StaticScene,
)


def float_values(values: np.ndarray, precision: int = 8) -> str:
    return " ".join(f"{float(value):.{precision}g}" for value in values.reshape(-1))


def int_values(values: np.ndarray) -> str:
    return " ".join(str(int(value)) for value in values.reshape(-1))


def static_scene_xml(scene: StaticScene) -> str:
    yaw = np.arctan2(scene.table_axis_x[1], scene.table_axis_x[0])
    table_quaternion = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    half_length = 0.5 * scene.table_length
    half_width = 0.5 * scene.table_width
    leg_inset = min(0.07, 0.22 * min(scene.table_length, scene.table_width))
    leg_half_width = 0.032
    leg_height = scene.table_top_z - scene.table_thickness
    if leg_height <= 0:
        raise ValueError("Tabletop is too low to create positive-height table legs")

    legs: list[str] = []
    for local_x in (-half_length + leg_inset, half_length - leg_inset):
        for local_y in (-half_width + leg_inset, half_width - leg_inset):
            legs.append(
                f'<geom type="box" pos="{local_x:.9g} {local_y:.9g} '
                f'{0.5 * leg_height:.9g}" size="{leg_half_width:.9g} '
                f'{leg_half_width:.9g} {0.5 * leg_height:.9g}" '
                f'material="table_material"/>'
            )

    rim_radius = max(0.003, min(0.006, scene.cup_radius * 0.09))
    rim_height = 0.5 * scene.cup_height + 0.002
    rim_segments: list[str] = []
    segment_count = 20
    for segment in range(segment_count):
        angle_0 = 2.0 * np.pi * segment / segment_count
        angle_1 = 2.0 * np.pi * (segment + 1) / segment_count
        point_0 = np.array(
            [
                scene.cup_radius * np.cos(angle_0),
                scene.cup_radius * np.sin(angle_0),
                rim_height,
            ]
        )
        point_1 = np.array(
            [
                scene.cup_radius * np.cos(angle_1),
                scene.cup_radius * np.sin(angle_1),
                rim_height,
            ]
        )
        rim_segments.append(
            f'<geom type="capsule" fromto="{float_values(point_0)} '
            f'{float_values(point_1)}" size="{rim_radius:.9g}" '
            f'material="cup_material" mass="0" contype="0" conaffinity="0"/>'
        )

    return f"""
      <body name="static_table"
            pos="{scene.table_center_xy[0]:.9g} {scene.table_center_xy[1]:.9g} 0"
            quat="{float_values(table_quaternion)}">
        <geom name="tabletop" type="box"
              pos="0 0 {scene.table_top_z - 0.5 * scene.table_thickness:.9g}"
              size="{half_length:.9g} {half_width:.9g} {0.5 * scene.table_thickness:.9g}"
              material="table_material"/>
        {''.join(legs)}
      </body>
      <!-- A mocap body has no joint or dynamic inertial state.  Its zero-mass
           geoms retain visible/collision volume while following our trajectory. -->
      <body name="massless_cup" mocap="true"
            pos="{float_values(scene.cup_center)}">
        <geom name="cup_collision" type="cylinder"
              size="{scene.cup_radius:.9g} {0.5 * scene.cup_height:.9g}"
              material="cup_material" mass="0"/>
        <geom name="cup_opening" type="cylinder"
              pos="0 0 {0.5 * scene.cup_height + 0.0025:.9g}"
              size="{0.82 * scene.cup_radius:.9g} 0.0015"
              material="cup_inside_material" mass="0"
              contype="0" conaffinity="0"/>
        {''.join(rim_segments)}
      </body>
    """


def make_model(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    bind_positions: np.ndarray,
    fps: float,
    scene: StaticScene,
) -> mujoco.MjModel:
    if weights.shape != (len(vertices), len(bind_positions)):
        raise ValueError(
            f"weights must have shape {(len(vertices), len(bind_positions))}, "
            f"got {weights.shape}"
        )
    if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError("SMPL-X skin weights do not sum to one")

    bodies: list[str] = []
    bones: list[str] = []
    identity_quaternion = "1 0 0 0"
    for joint, bind_position in enumerate(bind_positions):
        body_name = f"smplx_joint_{joint:02d}"
        bodies.append(
            f'<body name="{body_name}" mocap="true" '
            f'pos="{float_values(bind_position)}"/>'
        )
        vertex_ids = np.flatnonzero(weights[:, joint] > 1e-8)
        vertex_weights = weights[vertex_ids, joint]
        bones.append(
            f'<bone body="{body_name}" bindpos="{float_values(bind_position)}" '
            f'bindquat="{identity_quaternion}" vertid="{int_values(vertex_ids)}" '
            f'vertweight="{float_values(vertex_weights)}"/>'
        )

    timestep = 1.0 / fps
    xml = f"""
    <mujoco model="moshpp_smplx_skin">
      <option timestep="{timestep:.12g}" gravity="0 0 -9.81"/>
      <statistic center="0 0 1" extent="2.1"/>
      <visual>
        <headlight ambient="0.45 0.45 0.45" diffuse="0.72 0.72 0.72"
                   specular="0.15 0.15 0.15"/>
        <rgba haze="0.12 0.15 0.19 1"/>
        <quality shadowsize="4096"/>
      </visual>
      <asset>
        <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.16 0.18 0.22" rgb2="0.28 0.31 0.36"/>
        <material name="floor_material" texture="floor_grid" texrepeat="8 8"
                  reflectance="0.12"/>
        <material name="skin_material" rgba="0.72 0.49 0.36 1"
                  specular="0.18" shininess="0.12"/>
        <material name="table_material" rgba="0.34 0.20 0.11 1"
                  specular="0.12" shininess="0.05"/>
        <material name="cup_material" rgba="0.14 0.42 0.78 1"
                  specular="0.3" shininess="0.2"/>
        <material name="cup_inside_material" rgba="0.035 0.045 0.065 1"
                  specular="0.05" shininess="0.02"/>
        <skin name="smplx_body" material="skin_material"
              vertex="{float_values(vertices)}"
              face="{int_values(faces)}">
          {''.join(bones)}
        </skin>
      </asset>
      <worldbody>
        <light pos="-2 -3 4" dir="0.35 0.5 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="2 1 3" dir="-0.5 -0.2 -1" diffuse="0.35 0.37 0.42"/>
        <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1"
              material="floor_material" contype="0" conaffinity="0"/>
        {static_scene_xml(scene)}
        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)
