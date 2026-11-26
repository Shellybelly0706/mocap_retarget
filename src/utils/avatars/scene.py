"""Reusable avatars scene helpers."""

from __future__ import annotations

from xml.sax.saxutils import escape

import mujoco
import numpy as np

from utils.avatars.assets import (
    PROJECT_ROOT,
)
from utils.avatars.marker_motion import marker_color
from utils.avatars.models import (
    AnatomicalPart,
    AvatarAsset,
    AvatarConfig,
)
from utils.avatars.obstacles import (
    FLOOR_THICKNESS,
)
from utils.avatars.pose import (
    matrices_to_mujoco_quaternions,
    segment_style,
)
from utils.avatars.scene_xml import (
    xml_values,
)

CHRISTMAS_CUPBOARD_WALL_TEXTURE = (
    PROJECT_ROOT / "assets" / "christmas" / "brick_windows_v2_flipped.png"
)


CHRISTMAS_WINDOW_WALL_TEXTURE = (
    PROJECT_ROOT / "assets" / "christmas" / "brick_windows_3x4.png"
)


CHRISTMAS_FURNITURE_TEXTURE = (
    PROJECT_ROOT / "assets" / "christmas" / "wood_furniture.png"
)


CHRISTMAS_FLOOR_TEXTURE = (
    PROJECT_ROOT / "assets" / "christmas" / "wood_floor_carpet.png"
)


def christmas_asset_xml(enabled: bool) -> str:
    """Return optional Christmas texture/material declarations."""
    if not enabled:
        return ""
    for texture in (
        CHRISTMAS_CUPBOARD_WALL_TEXTURE,
        CHRISTMAS_WINDOW_WALL_TEXTURE,
        CHRISTMAS_FURNITURE_TEXTURE,
        CHRISTMAS_FLOOR_TEXTURE,
    ):
        if not texture.is_file():
            raise FileNotFoundError(f"Christmas texture not found: {texture}")
    return f"""
        <texture name="christmas_cupboard_wall_texture" type="2d"
                 file="{escape(str(CHRISTMAS_CUPBOARD_WALL_TEXTURE))}"
                 content_type="image/png"/>
        <material name="christmas_cupboard_wall"
                  texture="christmas_cupboard_wall_texture"
                  texrepeat="1 1" texuniform="false"
                  specular="0.05" shininess="0.03"/>
        <texture name="christmas_window_wall_texture" type="2d"
                 file="{escape(str(CHRISTMAS_WINDOW_WALL_TEXTURE))}"
                 content_type="image/png"/>
        <material name="christmas_window_wall"
                  texture="christmas_window_wall_texture"
                  texrepeat="1 1" texuniform="false"
                  specular="0.05" shininess="0.03"/>
        <texture name="christmas_furniture_texture" type="2d"
                 file="{escape(str(CHRISTMAS_FURNITURE_TEXTURE))}"
                 content_type="image/png"/>
        <material name="christmas_furniture"
                  texture="christmas_furniture_texture"
                  texrepeat="2 2" texuniform="false"
                  specular="0.12" shininess="0.08"/>
        <texture name="christmas_floor_texture" type="2d"
                 file="{escape(str(CHRISTMAS_FLOOR_TEXTURE))}"
                 content_type="image/png"/>
        <material name="christmas_floor" texture="christmas_floor_texture"
                  texrepeat="1 1" texuniform="false"
                  specular="0.08" shininess="0.04"/>
    """


def build_upper_body_model(
    marker_names: list[str],
    marker_positions: np.ndarray,
    reconstructed: set[str],
    edges: list[tuple[str, str]],
    segment_midpoints: np.ndarray,
    segment_lengths: np.ndarray,
    anatomical_parts: list[AnatomicalPart],
    body_scale: float,
    marker_size: float,
    show_markers: bool,
    scene_xml: str,
    christmas: bool,
) -> mujoco.MjModel:
    bodies: list[str] = []
    ground_height = -FLOOR_THICKNESS - 0.001 if scene_xml else 0.0
    seasonal_assets = christmas_asset_xml(christmas)

    if show_markers:
        for index, (name, position) in enumerate(
            zip(marker_names, marker_positions, strict=True)
        ):
            pos = " ".join(f"{value:.9g}" for value in position)
            color = marker_color(name, reconstructed).split()
            rgba = " ".join([*color[:3], "0.65"])
            bodies.append(
                f"""<body name="marker_{index:02d}_{escape(name)}"
                           mocap="true" pos="{pos}">
                      <geom type="sphere" size="{marker_size:.9g}"
                            rgba="{rgba}" contype="0" conaffinity="0"/>
                    </body>"""
            )

    for index, ((parent, child), midpoint, length) in enumerate(
        zip(edges, segment_midpoints, segment_lengths, strict=True)
    ):
        pos = " ".join(f"{value:.9g}" for value in midpoint)
        nominal_radius, material = segment_style(parent, child, body_scale)
        radius = min(nominal_radius, length * 0.28)
        half_length = max(length * 0.5 - radius, length * 0.12)
        bodies.append(
            f"""<body name="segment_{index:02d}_{escape(parent)}_{escape(child)}"
                       mocap="true" pos="{pos}">
                  <geom type="capsule" size="{radius:.9g} {half_length:.9g}"
                        material="{material}"
                        contype="0" conaffinity="0"/>
                </body>"""
        )

    for part in anatomical_parts:
        pos = " ".join(f"{value:.9g}" for value in part.positions[0])
        size = " ".join(f"{value:.9g}" for value in part.size)
        bodies.append(
            f"""<body name="anatomy_{escape(part.name)}" mocap="true" pos="{pos}">
                  <geom type="{part.geom_type}" size="{size}" material="{part.material}"
                        contype="0" conaffinity="0"/>
                </body>"""
        )

    xml = f"""
    <mujoco model="optitrack_fitted_upper_body">
      <option timestep="0.008333333333" gravity="0 0 -9.81"/>
      <visual>
        <headlight ambient="0.38 0.38 0.38" diffuse="0.72 0.72 0.72"
                   specular="0.18 0.18 0.18"/>
        <rgba haze="0.15 0.2 0.25 1"/>
      </visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.18 0.18 0.18" rgb2="0.28 0.28 0.28"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
        <material name="skin" rgba="0.76 0.56 0.42 1"
                  specular="0.18" shininess="0.12"/>
        <material name="shirt" rgba="0.16 0.34 0.60 1"
                  specular="0.12" shininess="0.08"/>
        <material name="trousers" rgba="0.10 0.14 0.21 1"
                  specular="0.08" shininess="0.04"/>
        {seasonal_assets}
      </asset>
      <worldbody>
        <light pos="-2 -2 4" dir="0.4 0.4 -1" diffuse="0.75 0.75 0.75"/>
        <light pos="2 1 3" dir="-0.5 -0.2 -1" diffuse="0.35 0.38 0.42"/>
        <geom name="floor" type="plane" pos="0 0 {ground_height:.9g}"
              size="5 5 0.1" material="grid"/>
        {scene_xml}
        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def build_avatar_model(
    config: AvatarConfig,
    asset: AvatarAsset,
    avatar_scale: float,
    marker_names: list[str],
    marker_positions: np.ndarray,
    reconstructed: set[str],
    marker_size: float,
    show_markers: bool,
    scene_xml: str,
    christmas: bool,
) -> mujoco.MjModel:
    """Build a MuJoCo model containing a textured, skinned VRM avatar."""
    if not config.texture.is_file():
        raise FileNotFoundError(f"Avatar texture not found: {config.texture}")
    ground_height = -FLOOR_THICKNESS - 0.001 if scene_xml else 0.0
    seasonal_assets = christmas_asset_xml(christmas)

    bind_positions = asset.bind_positions * avatar_scale
    vertices = asset.vertices * avatar_scale
    bind_quaternions = matrices_to_mujoco_quaternions(asset.bind_rotations)
    avatar_bodies = []
    skin_bones = []
    for index, (position, quaternion) in enumerate(
        zip(bind_positions, bind_quaternions, strict=True)
    ):
        body_name = f"avatar_joint_{index:02d}"
        avatar_bodies.append(
            f"""<body name="{body_name}" mocap="true"
                       pos="{xml_values(position)}" quat="{xml_values(quaternion)}"/>"""
        )
        vertex_ids = np.flatnonzero(asset.weights[:, index] > 1e-6)
        vertex_weights = asset.weights[vertex_ids, index]
        skin_bones.append(
            f"""<bone body="{body_name}" bindpos="{xml_values(position)}"
                       bindquat="{xml_values(quaternion)}"
                       vertid="{xml_values(vertex_ids, precision=12)}"
                       vertweight="{xml_values(vertex_weights)}"/>"""
        )

    marker_bodies = []
    if show_markers:
        for index, (name, position) in enumerate(
            zip(marker_names, marker_positions, strict=True)
        ):
            color = marker_color(name, reconstructed).split()
            rgba = " ".join([*color[:3], "0.65"])
            marker_bodies.append(
                f"""<body name="marker_{index:02d}_{escape(name)}" mocap="true"
                           pos="{xml_values(position)}">
                      <geom type="sphere" size="{marker_size:.9g}" rgba="{rgba}"
                            contype="0" conaffinity="0"/>
                    </body>"""
            )

    xml = f"""
    <mujoco model="optitrack_{config.name.lower()}_avatar">
      <option timestep="0.008333333333" gravity="0 0 -9.81"/>
      <visual>
        <headlight ambient="0.42 0.42 0.42" diffuse="0.72 0.72 0.72"
                   specular="0.18 0.18 0.18"/>
        <rgba haze="0.15 0.2 0.25 1"/>
      </visual>
      <asset>
        <texture name="grid_texture" type="2d" builtin="checker"
                 width="512" height="512"
                 rgb1="0.18 0.18 0.18" rgb2="0.28 0.28 0.28"/>
        <material name="grid" texture="grid_texture" texrepeat="8 8"
                  reflectance="0.1"/>
        <texture name="avatar_texture" type="2d" file="{escape(str(config.texture))}"
                 content_type="image/png" vflip="false"/>
        <!-- The source avatars use unlit/MToon materials.  Emission preserves
             their painted colors instead of shading dark areas
             down to black under MuJoCo's directional lights. -->
        <material name="avatar_material" texture="avatar_texture"
                  emission="1" specular="0" shininess="0"/>
        {seasonal_assets}
        <skin name="{config.name.lower()}_upper_body" material="avatar_material"
              vertex="{xml_values(vertices)}"
              texcoord="{xml_values(asset.texcoords)}"
              face="{xml_values(asset.faces, precision=12)}">
          {''.join(skin_bones)}
        </skin>
      </asset>
      <worldbody>
        <light pos="-2 -2 4" dir="0.4 0.4 -1" diffuse="0.78 0.78 0.78"/>
        <light pos="2 1 3" dir="-0.5 -0.2 -1" diffuse="0.35 0.38 0.42"/>
        <geom name="floor" type="plane" pos="0 0 {ground_height:.9g}"
              size="5 5 0.1" material="grid"/>
        {scene_xml}
        {''.join(marker_bodies)}
        {''.join(avatar_bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)
