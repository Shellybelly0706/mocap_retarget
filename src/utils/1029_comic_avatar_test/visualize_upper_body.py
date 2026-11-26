#!/usr/bin/env python3
"""Retarget OptiTrack upper-body motion onto a textured human avatar.

The fitted body uses the solved global ``Bone`` frames exported by Motive.  It
is a kinematic visualization, not a physics-driven MuJoCo humanoid.  Use
``--avatar`` to select a local VRM avatar; ``--mannequin`` selects the earlier
procedural body.  This CSV does not contain solved leg bones.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from visualize_markers import (
    DEFAULT_CSV,
    load_marker_motion,
    marker_color,
    parse_groups,
)


SKIN_MATERIAL = "skin"
SHIRT_MATERIAL = "shirt"
TROUSERS_MATERIAL = "trousers"
FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


@dataclass(frozen=True)
class AnatomicalPart:
    name: str
    geom_type: str
    size: tuple[float, ...]
    material: str
    positions: np.ndarray
    quaternions: np.ndarray


@dataclass(frozen=True)
class AvatarAsset:
    vertices: np.ndarray
    texcoords: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    joint_names: list[str]
    bind_positions: np.ndarray
    bind_rotations: np.ndarray


@dataclass(frozen=True)
class AvatarConfig:
    name: str
    source: Path
    skin: Path
    texture: Path


@dataclass(frozen=True)
class SceneObstacles:
    """Static room geometry inferred from the start and end calibration poses."""

    xml: str
    table_corner: np.ndarray
    table_length: float
    board_corner: np.ndarray
    table_forward: np.ndarray
    wall_normal: np.ndarray
    wall_contact: np.ndarray
    wall_span: float
    measured_board_corner_height: float
    cylinder_positions: np.ndarray
    cylinder_pickup_frames: tuple[int, int]
    cylinder_release_frames: tuple[int, int]


AVATAR_CONFIGS = {
    "baldman": AvatarConfig(
        name="Baldman",
        source=PROJECT_ROOT / "assets" / "Baldman.vrm",
        skin=PROJECT_ROOT / "assets" / "baldman" / "upper_body_skin.npz",
        texture=PROJECT_ROOT / "assets" / "baldman" / "avatar_texture.png",
    ),
    "cookieman": AvatarConfig(
        name="Cookieman",
        source=PROJECT_ROOT / "assets" / "Cookieman.vrm",
        skin=PROJECT_ROOT / "assets" / "cookieman" / "upper_body_skin.npz",
        texture=PROJECT_ROOT / "assets" / "cookieman" / "avatar_texture.png",
    ),
}
DEFAULT_AVATAR = "baldman"

TABLE_HEIGHT = 0.74
BOARD_LENGTH = 0.75
BOARD_WIDTH = 0.60
BOARD_THICKNESS = 0.02
BOARD_BOTTOM_HEIGHT = 1.20
WALL_THICKNESS = 0.10
TABLE_WALL_CLEARANCE = 0.005
FLOOR_THICKNESS = 0.05
CYLINDER_RADIUS = 0.03
CYLINDER_HALF_HEIGHT = 0.06


def upper_body_edges() -> list[tuple[str, str]]:
    """Return visible limb links; torso and palms use fitted solid volumes."""
    edges: list[tuple[str, str]] = []
    for side in ("L", "R"):
        edges.extend(
            [
                ("Chest", f"{side}Shoulder"),
                (f"{side}Shoulder", f"{side}UArm"),
                (f"{side}UArm", f"{side}FArm"),
                (f"{side}FArm", f"{side}Hand"),
            ]
        )
        for finger in FINGER_NAMES:
            edges.extend(
                [
                    (f"{side}{finger}1", f"{side}{finger}2"),
                    (f"{side}{finger}2", f"{side}{finger}3"),
                ]
            )
    return edges


def load_bones(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(7)]
    groups = parse_groups(header)
    table = pd.read_csv(csv_path, skiprows=7, header=None, na_values=[""])

    result: dict[str, np.ndarray] = {}
    for group in groups:
        if group["type"] != "Bone":
            continue
        name = str(group["name"]).split(":", 1)[-1]
        values = table.iloc[
            :, int(group["start"]) : int(group["end"])
        ].to_numpy(float)
        if values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError(f"Bone {name!r} does not contain complete XYZW + XYZ data")
        result[name] = values
    return result


def quaternions_aligning_z(vectors: np.ndarray) -> np.ndarray:
    """Return WXYZ quaternions rotating local +Z onto each input vector."""
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("Cannot orient a zero-length body segment")
    unit = vectors / lengths[:, None]

    # XYZW quaternion rotating a=(0,0,1) to b=unit:
    # vector=cross(a,b)=(-by,bx,0), scalar=1+dot(a,b)=1+bz.
    xyzw = np.column_stack(
        (-unit[:, 1], unit[:, 0], np.zeros(len(unit)), 1.0 + unit[:, 2])
    )
    opposite = np.linalg.norm(xyzw, axis=1) < 1e-8
    xyzw[opposite] = np.array([1.0, 0.0, 0.0, 0.0])
    xyzw /= np.linalg.norm(xyzw, axis=1, keepdims=True)
    return xyzw[:, [3, 0, 1, 2]]  # MuJoCo order WXYZ.


def segment_style(parent: str, child: str, scale: float) -> tuple[float, str]:
    """Return an anatomical radius and material for a skeleton link."""
    if any(token in child for token in FINGER_NAMES):
        return 0.008 * scale, SKIN_MATERIAL
    if child.endswith("Shoulder"):
        return 0.042 * scale, SHIRT_MATERIAL
    if child.endswith("UArm"):
        return 0.052 * scale, SHIRT_MATERIAL
    if child.endswith("FArm"):
        return 0.041 * scale, SKIN_MATERIAL
    if child.endswith("Hand"):
        return 0.033 * scale, SKIN_MATERIAL
    return 0.025 * scale, SKIN_MATERIAL


def prepare_segments(
    bones: dict[str, np.ndarray], edges: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint_tracks = []
    quaternion_tracks = []
    lengths = []
    for parent, child in edges:
        if parent not in bones or child not in bones:
            raise KeyError(f"Required upper-body edge {parent}->{child} is unavailable")
        start = bones[parent][:, 4:7]
        end = bones[child][:, 4:7]
        vector = end - start
        midpoint_tracks.append((start + end) * 0.5)
        quaternion_tracks.append(quaternions_aligning_z(vector))
        lengths.append(float(np.median(np.linalg.norm(vector, axis=1))))
    return (
        np.stack(midpoint_tracks, axis=1),
        np.stack(quaternion_tracks, axis=1),
        np.asarray(lengths),
    )


def bone_quaternions(bones: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Convert an OptiTrack XYZW quaternion track to MuJoCo WXYZ order."""
    return bones[name][:, :4][:, [3, 0, 1, 2]]


def median_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(first - second, axis=1)))


def load_avatar_asset(config: AvatarConfig) -> AvatarAsset:
    path = config.skin
    if not path.is_file():
        raise FileNotFoundError(
            f"Avatar asset not found: {path}. Run src/utils/1029_comic_avatar_test/convert_vrm_avatar.py "
            f"{config.source} {config.skin} --upper-body to generate it."
        )
    with np.load(path, allow_pickle=False) as archive:
        return AvatarAsset(
            vertices=archive["vertices"],
            texcoords=archive["texcoords"],
            faces=archive["faces"],
            weights=archive["weights"],
            joint_names=archive["joint_names"].tolist(),
            bind_positions=archive["bind_positions"],
            bind_rotations=archive["bind_rotations"],
        )


def normalized(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Cannot construct an avatar frame from coincident joints")
    return vectors / lengths


def anatomical_frames(
    lateral: np.ndarray, up: np.ndarray, front_hint: np.ndarray | None = None
) -> np.ndarray:
    """Return right-handed frames with columns front, left, and up."""
    lateral = normalized(np.atleast_2d(lateral))
    up = normalized(np.atleast_2d(up))
    lateral = normalized(lateral - np.sum(lateral * up, axis=1, keepdims=True) * up)
    if front_hint is None:
        front = normalized(np.cross(lateral, up))
    else:
        front = np.atleast_2d(front_hint)
        front = normalized(front - np.sum(front * up, axis=1, keepdims=True) * up)
    lateral = normalized(np.cross(up, front))
    return np.stack((front, lateral, up), axis=-1)


def limb_frames(direction: np.ndarray, up_reference: np.ndarray) -> np.ndarray:
    """Return frames whose first axis follows a limb and whose roll stays upright."""
    axis = normalized(np.atleast_2d(direction))
    up_reference = normalized(np.atleast_2d(up_reference))
    side = normalized(np.cross(up_reference, axis))
    local_up = normalized(np.cross(axis, side))
    return np.stack((axis, side, local_up), axis=-1)


def palm_frames(
    wrist: np.ndarray,
    index_base: np.ndarray,
    middle_base: np.ndarray,
    little_base: np.ndarray,
) -> np.ndarray:
    """Return stable palm frames with forward, across-palm, and normal axes."""
    forward = normalized(np.atleast_2d(middle_base - wrist))
    across = normalized(np.atleast_2d(index_base - little_base))
    normal = normalized(np.cross(across, forward))
    across = normalized(np.cross(forward, normal))
    return np.stack((forward, across, normal), axis=-1)


def matrices_to_mujoco_quaternions(matrices: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    return xyzw[:, [3, 0, 1, 2]].reshape(*matrices.shape[:-2], 4)


def horizontal_body_axes(
    bones: dict[str, np.ndarray], frame_slice: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal forward and left axes from a stable pose window."""
    left = np.median(
        bones["LShoulder"][frame_slice, 4:7]
        - bones["RShoulder"][frame_slice, 4:7],
        axis=0,
    )
    left[2] = 0.0
    left /= np.linalg.norm(left)
    forward = np.cross(left, np.array([0.0, 0.0, 1.0]))
    forward /= np.linalg.norm(forward)
    return forward, left


def snap_horizontal_axes_to_grid(
    measured_forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap a horizontal heading to its nearest checkerboard/grid axis."""
    forward = np.zeros(3)
    dominant_axis = int(np.argmax(np.abs(measured_forward[:2])))
    forward[dominant_axis] = np.copysign(1.0, measured_forward[dominant_axis])
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    return forward, left


def static_box_xml(
    name: str,
    center: np.ndarray,
    half_size: np.ndarray,
    axes: np.ndarray,
    rgba: str | None,
    material: str | None = None,
) -> str:
    """Return one oriented, world-fixed MuJoCo box geom."""
    if (rgba is None) == (material is None):
        raise ValueError("A static box requires exactly one of rgba or material")
    quaternion = matrices_to_mujoco_quaternions(np.asarray(axes))
    appearance = (
        f'rgba="{rgba}"' if rgba is not None else f'material="{escape(material)}"'
    )
    return (
        f'<geom name="{escape(name)}" type="box" '
        f'pos="{xml_values(center)}" size="{xml_values(half_size)}" '
        f'quat="{xml_values(quaternion)}" {appearance}/>'
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
    return f'''
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
    '''


def carried_object_track(
    grip_positions: np.ndarray,
    initial_position: np.ndarray,
    final_position: np.ndarray,
    pickup_frame: int,
    release_frame: int,
    transition_frames: int = 30,
) -> np.ndarray:
    """Create a stationary-carry-stationary object trajectory."""
    track = np.repeat(initial_position[None, :], len(grip_positions), axis=0)
    pickup_end = min(pickup_frame + transition_frames, release_frame)
    release_start = max(pickup_end, release_frame - transition_frames)

    if pickup_end > pickup_frame:
        alpha = np.linspace(0.0, 1.0, pickup_end - pickup_frame, endpoint=False)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        track[pickup_frame:pickup_end] = (
            (1.0 - alpha[:, None]) * initial_position
            + alpha[:, None] * grip_positions[pickup_frame:pickup_end]
        )
    track[pickup_end:release_start] = grip_positions[pickup_end:release_start]
    if release_frame > release_start:
        alpha = np.linspace(0.0, 1.0, release_frame - release_start, endpoint=False)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        track[release_start:release_frame] = (
            (1.0 - alpha[:, None]) * grip_positions[release_start:release_frame]
            + alpha[:, None] * final_position
        )
    track[release_frame:] = final_position
    return track


def reconstruct_scene_obstacles(
    bones: dict[str, np.ndarray],
    frame_rate: float,
    table_length: float | None,
    table_width: float,
    wall_width: float,
    wall_height: float,
    christmas: bool = False,
) -> SceneObstacles:
    """Infer two tables, a wall, and its board from the calibration gestures.

    The first stable second supplies the near-right table corner and table axes.
    The last stable second supplies the wall-facing direction and the board's
    lower-left corner. The tables and wall are snapped to the nearest floor-grid
    axes. Unless explicitly overridden, both end-to-end tables are shortened so
    the second table's complete end face sits beside the wall with a small
    non-intersection clearance.
    """
    stable_frames = max(1, min(len(bones["RHand"]), round(frame_rate)))
    start_slice = slice(0, stable_frames)
    end_slice = slice(-stable_frames, None)
    measured_table_forward, _ = horizontal_body_axes(bones, start_slice)
    table_forward, table_left = snap_horizontal_axes_to_grid(
        measured_table_forward
    )
    measured_wall_normal, _ = horizontal_body_axes(bones, end_slice)
    wall_normal, wall_left = snap_horizontal_axes_to_grid(
        measured_wall_normal
    )

    table_corner = np.median(bones["RHand"][start_slice, 4:7], axis=0)
    table_corner[2] = TABLE_HEIGHT
    root_name = next(iter(bones))
    avatar_start = np.median(bones[root_name][start_slice, 4:7], axis=0)
    table_width_direction = table_left.copy()
    if np.dot(table_width_direction, table_corner - avatar_start) < 0.0:
        table_width_direction *= -1.0
    measured_board_corner = np.median(
        bones["LHand"][end_slice, 4:7], axis=0
    )
    board_corner = measured_board_corner.copy()
    board_corner[2] = BOARD_BOTTOM_HEIGHT
    up = np.array([0.0, 0.0, 1.0])

    # The hand indicates the shelf's outer corner. The wall surface is one
    # board-width farther along the performer's forward direction.
    wall_axes = np.column_stack((wall_left, -wall_normal, up))
    wall_surface = board_corner + wall_normal * BOARD_WIDTH

    if table_length is None:
        alignment = float(np.dot(table_forward, wall_normal))
        if abs(alignment) < 1e-9:
            raise ValueError(
                "The grid-aligned table direction does not meet the wall plane"
            )
        normal_distance = float(
            np.dot(wall_surface - table_corner, wall_normal)
        )
        adjusted_distance = normal_distance - np.copysign(
            TABLE_WALL_CLEARANCE, normal_distance
        )
        combined_length = adjusted_distance / alignment
        if combined_length <= 0.0:
            raise ValueError("The inferred wall is behind the table corner")
        table_length = combined_length * 0.5

    # Put the width direction first so (-left, forward, up) remains a proper
    # right-handed rotation when the table extends away from the avatar.
    table_axes = np.column_stack((table_width_direction, table_forward, up))
    table_half_size = np.array(
        [table_width * 0.5, table_length * 0.5, TABLE_HEIGHT * 0.5]
    )
    boxes = []
    for index in range(2):
        center = (
            table_corner
            + table_forward * (index + 0.5) * table_length
            + table_width_direction * table_width * 0.5
            - up * TABLE_HEIGHT * 0.5
        )
        boxes.append(
            static_box_xml(
                f"table_{index + 1}",
                center,
                table_half_size,
                table_axes,
                None if christmas else "0.46 0.27 0.12 1",
                material="christmas_furniture" if christmas else None,
            )
        )

    # Extend the table-side end of the wall to the first table boundary it
    # meets. This creates a corner without letting the wall pass through the
    # brown table geometry. The opposite endpoint retains the requested
    # minimum wall width.
    table_base_corner = table_corner.copy()
    table_base_corner[2] = 0.0
    table_corners = (
        table_base_corner,
        table_base_corner + table_forward * (2.0 * table_length),
        table_base_corner
        + table_forward * (2.0 * table_length)
        + table_width_direction * table_width,
        table_base_corner + table_width_direction * table_width,
    )
    intersections = []
    for first, second in zip(
        table_corners, (*table_corners[1:], table_corners[0]), strict=True
    ):
        edge = second - first
        denominator = float(np.dot(edge, wall_normal))
        if abs(denominator) < 1e-9:
            continue
        fraction = float(np.dot(wall_surface - first, wall_normal) / denominator)
        if -1e-9 <= fraction <= 1.0 + 1e-9:
            intersections.append(first + np.clip(fraction, 0.0, 1.0) * edge)
    if intersections:
        wall_contact = min(
            intersections,
            key=lambda point: float(np.dot(point - wall_surface, wall_left)),
        )
    else:
        distances = [
            abs(float(np.dot(point - wall_surface, wall_normal)))
            for point in table_corners
        ]
        closest_distance = min(distances)
        closest_corners = [
            point
            for point, distance in zip(table_corners, distances, strict=True)
            if abs(distance - closest_distance) < 1e-8
        ]
        wall_contact = min(
            closest_corners,
            key=lambda point: float(np.dot(point - wall_surface, wall_left)),
        )
    contact_offset = float(np.dot(wall_contact - wall_surface, wall_left))
    return_wall_offset = contact_offset - TABLE_WALL_CLEARANCE
    lower_offset = min(-wall_width * 0.5, return_wall_offset)
    upper_offset = max(wall_width * 0.5, return_wall_offset)
    wall_span = upper_offset - lower_offset
    wall_surface_center = wall_surface + wall_left * (
        0.5 * (lower_offset + upper_offset)
    )
    wall_center = (
        wall_surface_center
        + wall_normal * WALL_THICKNESS * 0.5
        + up * (wall_height * 0.5 - BOARD_BOTTOM_HEIGHT)
    )
    boxes.append(
        static_box_xml(
            "wall",
            wall_center,
            np.array([wall_span * 0.5, WALL_THICKNESS * 0.5, wall_height * 0.5]),
            wall_axes,
            None if christmas else "0.72 0.74 0.77 1",
            material="christmas_cupboard_wall" if christmas else None,
        )
    )

    # Wrap a second wall around the table's outer-width side. Both table faces
    # retain the same small clearance from their neighboring wall surfaces.
    return_corner = wall_surface + wall_left * return_wall_offset
    return_wall_axes = np.column_stack((-wall_normal, -wall_left, up))
    return_wall_center = (
        return_corner
        - wall_normal * wall_width * 0.5
        - wall_left * WALL_THICKNESS * 0.5
        + up * (wall_height * 0.5 - BOARD_BOTTOM_HEIGHT)
    )
    boxes.append(
        static_box_xml(
            "wall_return",
            return_wall_center,
            np.array([wall_width * 0.5, WALL_THICKNESS * 0.5, wall_height * 0.5]),
            return_wall_axes,
            None if christmas else "0.72 0.74 0.77 1",
            material="christmas_window_wall" if christmas else None,
        )
    )

    # Add a solid floor slab inside the L-shaped walls. The checker plane is
    # moved just below it by the model builders, avoiding coplanar flicker.
    # The return wall can lie at either end of the main wall depending on the
    # performer's final facing direction.  Extend the floor toward the farther
    # main-wall endpoint instead of assuming that endpoint is +wall_left.
    opposite_offset = (
        upper_offset
        if abs(upper_offset - return_wall_offset)
        >= abs(lower_offset - return_wall_offset)
        else lower_offset
    )
    floor_direction = np.copysign(1.0, opposite_offset - return_wall_offset)
    floor_span_along_wall = (
        abs(opposite_offset - return_wall_offset) + WALL_THICKNESS
    )
    floor_span_across_wall = wall_width + WALL_THICKNESS
    floor_corner = (
        return_corner
        + wall_normal * WALL_THICKNESS
        - wall_left * floor_direction * WALL_THICKNESS
    )
    floor_corner[2] = 0.0
    floor_center = (
        floor_corner
        - wall_normal * floor_span_across_wall * 0.5
        + wall_left * floor_direction * floor_span_along_wall * 0.5
        - up * FLOOR_THICKNESS * 0.5
    )
    floor_quaternion = matrices_to_mujoco_quaternions(return_wall_axes)
    floor_appearance = (
        'material="christmas_floor"'
        if christmas
        else 'rgba="0.32 0.32 0.32 1"'
    )
    boxes.append(
        f'<geom name="room_floor" type="box" '
        f'pos="{xml_values(floor_center)}" '
        f'size="{xml_values(np.array([floor_span_across_wall * 0.5, floor_span_along_wall * 0.5, FLOOR_THICKNESS * 0.5]))}" '
        f'quat="{xml_values(floor_quaternion)}" {floor_appearance}/>'
    )

    # The left hand marks the shelf's outer lower-left corner. The horizontal
    # board extends toward the performer's right and forward to meet the wall.
    board_axes = np.column_stack((-wall_left, wall_normal, up))
    board_center = (
        board_corner
        - wall_left * BOARD_LENGTH * 0.5
        + wall_normal * BOARD_WIDTH * 0.5
        + up * BOARD_THICKNESS * 0.5
    )
    boxes.append(
        static_box_xml(
            "wall_board",
            board_center,
            np.array(
                [BOARD_LENGTH * 0.5, BOARD_WIDTH * 0.5, BOARD_THICKNESS * 0.5]
            ),
            board_axes,
            None if christmas else "0.12 0.34 0.24 1",
            material="christmas_furniture" if christmas else None,
        )
    )

    # Reconstruct the two recorded right-hand transports. The cylinders begin
    # upright on the table, follow the fingertip grip, and settle upright on
    # the horizontal green shelf.
    grip_positions = np.mean(
        np.stack(
            [
                bones[name][:, 4:7]
                for name in ("RThumb3", "RIndex3", "RMiddle3")
            ],
            axis=0,
        ),
        axis=0,
    )
    frame_count = len(grip_positions)
    pickup_frames = tuple(
        min(frame_count - 1, round(seconds * frame_rate))
        for seconds in (2.58, 16.92)
    )
    release_frames = tuple(
        min(frame_count - 1, round(seconds * frame_rate))
        for seconds in (11.58, 23.08)
    )
    cylinder_initial = []
    for forward_offset in (0.08, 0.22):
        position = (
            table_corner
            + table_forward * forward_offset
            + table_width_direction * 0.16
            + up * CYLINDER_HALF_HEIGHT
        )
        cylinder_initial.append(position)

    shelf_right = -wall_left
    cylinder_final = []
    for release_frame in release_frames:
        sample_start = max(0, release_frame - 30)
        indicated = np.median(
            grip_positions[sample_start:release_frame], axis=0
        )
        shelf_depth = float(
            np.clip(
                np.dot(indicated - board_corner, wall_normal),
                CYLINDER_RADIUS + 0.02,
                BOARD_WIDTH - CYLINDER_RADIUS - 0.02,
            )
        )
        shelf_across = float(
            np.clip(
                np.dot(indicated - board_corner, shelf_right),
                CYLINDER_RADIUS + 0.02,
                BOARD_LENGTH - CYLINDER_RADIUS - 0.02,
            )
        )
        cylinder_final.append(
            board_corner
            + wall_normal * shelf_depth
            + shelf_right * shelf_across
            + up * (BOARD_THICKNESS + CYLINDER_HALF_HEIGHT)
        )

    cylinder_positions = np.stack(
        [
            carried_object_track(
                grip_positions,
                initial,
                final,
                pickup,
                release,
            )
            for initial, final, pickup, release in zip(
                cylinder_initial,
                cylinder_final,
                pickup_frames,
                release_frames,
                strict=True,
            )
        ],
        axis=1,
    )
    cylinder_colors = ("0.95 0.22 0.08 1", "0.12 0.48 0.95 1")
    for index, (position, color) in enumerate(
        zip(cylinder_positions[0], cylinder_colors, strict=True), start=1
    ):
        if christmas:
            boxes.append(
                f'<body name="carried_cylinder_{index}" pos="{xml_values(position)}">'
                f'<geom name="wine_glass_{index}_base" type="cylinder" '
                'pos="0 0 -0.055" size="0.05 0.005" '
                'rgba="0.88 0.94 1 0.38" contype="0" conaffinity="0"/>'
                f'<geom name="wine_glass_{index}_stem" type="cylinder" '
                'pos="0 0 -0.01" size="0.007 0.04" '
                'rgba="0.88 0.94 1 0.42" contype="0" conaffinity="0"/>'
                f'<geom name="wine_glass_{index}_bowl" type="ellipsoid" '
                'pos="0 0 0.095" size="0.055 0.055 0.07" '
                'rgba="0.88 0.94 1 0.28" contype="0" conaffinity="0"/>'
                f'<geom name="wine_{index}" type="cylinder" '
                'pos="0 0 0.085" size="0.043 0.04" '
                'rgba="0.48 0.015 0.035 0.82"/>'
                '</body>'
            )
        else:
            boxes.append(
                f'<body name="carried_cylinder_{index}" pos="{xml_values(position)}">'
                f'<geom name="cylinder_{index}" type="cylinder" '
                f'size="{CYLINDER_RADIUS:.9g} {CYLINDER_HALF_HEIGHT:.9g}" '
                f'rgba="{color}"/></body>'
            )

    return SceneObstacles(
        xml="\n".join(boxes),
        table_corner=table_corner,
        table_length=table_length,
        board_corner=board_corner,
        table_forward=table_forward,
        wall_normal=wall_normal,
        wall_contact=wall_contact,
        wall_span=wall_span,
        measured_board_corner_height=float(measured_board_corner[2]),
        cylinder_positions=cylinder_positions,
        cylinder_pickup_frames=pickup_frames,
        cylinder_release_frames=release_frames,
    )


def prepare_avatar_pose(
    asset: AvatarAsset,
    bones: dict[str, np.ndarray],
    marker_names: list[str],
    markers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retarget VRM humanoid joints onto the solved OptiTrack bone centers."""
    marker = {name: markers[:, index] for index, name in enumerate(marker_names)}
    required_markers = {"HeadTop", "HeadFront"}
    missing_markers = sorted(required_markers.difference(marker))
    if missing_markers:
        raise KeyError(
            f"Required avatar-fitting markers are unavailable: {missing_markers}"
        )

    joint_index = {name: index for index, name in enumerate(asset.joint_names)}
    vrm_fingers = {
        "Thumb": "Thumb",
        "Index": "Index",
        "Middle": "Middle",
        "Ring": "Ring",
        "Little": "Pinky",
    }
    stages = ("Proximal", "Intermediate", "Distal")
    root_name = next(iter(bones))
    required_bones = {
        root_name,
        "Ab",
        "Chest",
        "Neck",
        "Head",
        "LShoulder",
        "LUArm",
        "LFArm",
        "LHand",
        "RShoulder",
        "RUArm",
        "RFArm",
        "RHand",
    }
    for side_word, side_code in (("left", "L"), ("right", "R")):
        for vrm_finger, optitrack_finger in vrm_fingers.items():
            for segment, stage in enumerate(stages, start=1):
                if f"{side_word}{vrm_finger}{stage}" in joint_index:
                    required_bones.add(f"{side_code}{optitrack_finger}{segment}")
    missing_bones = sorted(required_bones.difference(bones))
    if missing_bones:
        raise KeyError(f"Required solved bones are unavailable: {missing_bones}")

    bone_position = {name: values[:, 4:7] for name, values in bones.items()}
    target: dict[str, np.ndarray] = {
        "hips": bone_position[root_name],
        "spine": bone_position["Ab"],
        "chest": 0.52 * bone_position["Ab"] + 0.48 * bone_position["Chest"],
        "upperChest": bone_position["Chest"],
        "neck": bone_position["Neck"],
        "head": bone_position["Head"],
        "leftShoulder": bone_position["LShoulder"],
        "leftUpperArm": bone_position["LUArm"],
        "leftLowerArm": bone_position["LFArm"],
        "leftHand": bone_position["LHand"],
        "rightShoulder": bone_position["RShoulder"],
        "rightUpperArm": bone_position["RUArm"],
        "rightLowerArm": bone_position["RFArm"],
        "rightHand": bone_position["RHand"],
    }
    for side_word, side_code in (("left", "L"), ("right", "R")):
        for vrm_finger, optitrack_finger in vrm_fingers.items():
            for segment, stage in enumerate(stages, start=1):
                joint_name = f"{side_word}{vrm_finger}{stage}"
                if joint_name in joint_index:
                    target[joint_name] = bone_position[
                        f"{side_code}{optitrack_finger}{segment}"
                    ]

    static_leg_joints = {
        "leftUpperLeg",
        "leftLowerLeg",
        "leftFoot",
        "leftToes",
        "rightUpperLeg",
        "rightLowerLeg",
        "rightFoot",
        "rightToes",
    }
    unsupported = sorted(
        set(asset.joint_names).difference(target).difference(static_leg_joints)
    )
    if unsupported:
        raise ValueError(f"No OptiTrack mapping for avatar joints: {unsupported}")

    source = asset.bind_positions
    source_position = {
        name: source[index] for name, index in joint_index.items()
    }
    source_up = source_position["head"] - source_position["hips"]
    source_body_frame = anatomical_frames(
        source_position["leftShoulder"] - source_position["rightShoulder"],
        source_up,
    )[0]
    target_up = target["head"] - target["hips"]
    target_body_frame = anatomical_frames(
        target["leftShoulder"] - target["rightShoulder"], target_up
    )
    body_deformation = target_body_frame @ source_body_frame.T

    target_head_frame = anatomical_frames(
        target["leftShoulder"] - target["rightShoulder"],
        marker["HeadTop"] - target["head"],
        marker["HeadFront"] - target["head"],
    )
    head_deformation = target_head_frame @ source_body_frame.T

    deformations = np.repeat(body_deformation[:, None, :, :], len(source), axis=1)
    deformations[:, joint_index["head"]] = head_deformation

    for side in ("left", "right"):
        chain = (
            f"{side}Shoulder",
            f"{side}UpperArm",
            f"{side}LowerArm",
        )
        children = (
            f"{side}UpperArm",
            f"{side}LowerArm",
            f"{side}Hand",
        )
        for name, child in zip(chain, children, strict=True):
            source_frame = limb_frames(
                source_position[child] - source_position[name], source_up
            )[0]
            target_frame = limb_frames(target[child] - target[name], target_up)
            deformations[:, joint_index[name]] = target_frame @ source_frame.T

        hand = f"{side}Hand"
        lower_arm = f"{side}LowerArm"
        source_hand_frame = limb_frames(
            source_position[hand] - source_position[lower_arm], source_up
        )[0]
        target_hand_frame = limb_frames(
            target[hand] - target[lower_arm], target_up
        )
        deformations[:, joint_index[hand]] = (
            target_hand_frame @ source_hand_frame.T
        )

        index = f"{side}IndexProximal"
        middle = f"{side}MiddleProximal"
        little = f"{side}LittleProximal"
        palm_joints = (index, middle, little)
        if not all(name in joint_index for name in palm_joints):
            continue
        source_palm = palm_frames(
            source_position[hand],
            source_position[index],
            source_position[middle],
            source_position[little],
        )[0]
        target_palm = palm_frames(
            target[hand], target[index], target[middle], target[little]
        )
        deformations[:, joint_index[hand]] = target_palm @ source_palm.T

        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            finger_joints = [f"{side}{finger}{stage}" for stage in stages]
            if not all(name in joint_index for name in finger_joints):
                continue
            for segment, name in enumerate(finger_joints):
                if segment < 2:
                    direction_name = finger_joints[segment + 1]
                    source_direction = (
                        source_position[direction_name] - source_position[name]
                    )
                    target_direction = target[direction_name] - target[name]
                else:
                    previous = finger_joints[segment - 1]
                    source_direction = source_position[name] - source_position[previous]
                    target_direction = target[name] - target[previous]
                source_frame = limb_frames(source_direction, source_palm[:, 2])[0]
                target_frame = limb_frames(target_direction, target_palm[:, :, 2])
                deformations[:, joint_index[name]] = target_frame @ source_frame.T

    # Use Motive's solved bone orientations for relative motion, including
    # twist around each bone axis.  The direction-based frame at the first
    # capture frame supplies the avatar-to-capture alignment; subsequent
    # frames apply the exact solved orientation change.
    joint_to_bone = {
        "hips": root_name,
        "spine": "Ab",
        "chest": "Ab",
        "upperChest": "Chest",
        "neck": "Neck",
        "head": "Head",
        "leftShoulder": "LShoulder",
        "leftUpperArm": "LUArm",
        "leftLowerArm": "LFArm",
        "leftHand": "LHand",
        "rightShoulder": "RShoulder",
        "rightUpperArm": "RUArm",
        "rightLowerArm": "RFArm",
        "rightHand": "RHand",
    }
    for side_word, side_code in (("left", "L"), ("right", "R")):
        for vrm_finger, optitrack_finger in vrm_fingers.items():
            for segment, stage in enumerate(stages, start=1):
                joint_to_bone[f"{side_word}{vrm_finger}{stage}"] = (
                    f"{side_code}{optitrack_finger}{segment}"
                )
    for name, bone_name in joint_to_bone.items():
        if name not in joint_index:
            continue
        solved_rotation = Rotation.from_quat(bones[bone_name][:, :4]).as_matrix()
        relative_rotation = solved_rotation @ solved_rotation[0].T
        first_alignment = deformations[0, joint_index[name]]
        deformations[:, joint_index[name]] = relative_rotation @ first_alignment

    rotations = deformations @ asset.bind_rotations[None, :, :, :]
    quaternions = matrices_to_mujoco_quaternions(rotations)

    source_height = float(
        asset.vertices[:, 2].max() - source_position["hips"][2]
    )
    target_height = float(
        np.median(np.linalg.norm(marker["HeadTop"] - target["hips"], axis=1))
    )
    avatar_scale = float(np.clip(target_height / source_height, 0.6, 1.4))

    # Preserve the avatar's own proportions.  The capture joint centers guide
    # bone directions, while forward kinematics keeps the avatar's bind-pose
    # bone lengths instead of stretching the skin to the subject's skeleton.
    parents = {
        "spine": "hips",
        "chest": "spine",
        "upperChest": "chest",
        "neck": "upperChest",
        "head": "neck",
        "leftUpperLeg": "hips",
        "leftLowerLeg": "leftUpperLeg",
        "leftFoot": "leftLowerLeg",
        "leftToes": "leftFoot",
        "rightUpperLeg": "hips",
        "rightLowerLeg": "rightUpperLeg",
        "rightFoot": "rightLowerLeg",
        "rightToes": "rightFoot",
        "leftShoulder": "upperChest",
        "leftUpperArm": "leftShoulder",
        "leftLowerArm": "leftUpperArm",
        "leftHand": "leftLowerArm",
        "rightShoulder": "upperChest",
        "rightUpperArm": "rightShoulder",
        "rightLowerArm": "rightUpperArm",
        "rightHand": "rightLowerArm",
    }
    for side in ("left", "right"):
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            proximal = f"{side}{finger}Proximal"
            intermediate = f"{side}{finger}Intermediate"
            distal = f"{side}{finger}Distal"
            parents[proximal] = f"{side}Hand"
            parents[intermediate] = proximal
            parents[distal] = intermediate

    positions = np.empty(
        (markers.shape[0], len(asset.joint_names), 3), dtype=np.float64
    )
    positions[:, joint_index["hips"]] = target["hips"]
    for name in asset.joint_names:
        if name == "hips":
            continue
        parent = parents[name]
        parent_index = joint_index[parent]
        child_index = joint_index[name]
        bind_offset = source[child_index] - source[parent_index]
        local_offset = asset.bind_rotations[parent_index].T @ bind_offset
        positions[:, child_index] = positions[:, parent_index] + np.einsum(
            "nij,j->ni",
            rotations[:, parent_index],
            local_offset * avatar_scale,
        )
    return positions, quaternions, avatar_scale


def prepare_anatomical_parts(
    bones: dict[str, np.ndarray],
    marker_names: list[str],
    markers: np.ndarray,
    root_name: str,
) -> tuple[list[AnatomicalPart], float]:
    """Fit smooth torso, head, neck, and palm volumes to the capture."""
    marker = {name: markers[:, index] for index, name in enumerate(marker_names)}
    required_markers = {
        "HeadTop",
        "LShoulderTop",
        "RShoulderTop",
        "LShoulderBack",
        "RShoulderBack",
        "WaistLFront",
        "WaistRFront",
        "WaistLBack",
        "WaistRBack",
    }
    missing = sorted(required_markers.difference(marker))
    if missing:
        raise KeyError(f"Required body-shape markers are unavailable: {missing}")

    shoulder_width = 0.5 * (
        median_distance(marker["LShoulderTop"], marker["RShoulderTop"])
        + median_distance(marker["LShoulderBack"], marker["RShoulderBack"])
    )
    waist_width = 0.5 * (
        median_distance(marker["WaistLFront"], marker["WaistRFront"])
        + median_distance(marker["WaistLBack"], marker["WaistRBack"])
    )
    waist_depth = 0.5 * (
        median_distance(marker["WaistLFront"], marker["WaistLBack"])
        + median_distance(marker["WaistRFront"], marker["WaistRBack"])
    )
    scale = float(np.clip(shoulder_width / 0.35, 0.75, 1.35))

    root = bones[root_name][:, 4:7]
    abdomen = bones["Ab"][:, 4:7]
    chest = bones["Chest"][:, 4:7]
    neck = bones["Neck"][:, 4:7]
    head_origin = bones["Head"][:, 4:7]

    root_ab_length = median_distance(root, abdomen)
    ab_chest_length = median_distance(abdomen, chest)
    chest_neck_length = median_distance(chest, neck)
    neck_head_vector = head_origin - neck
    neck_head_length = float(np.median(np.linalg.norm(neck_head_vector, axis=1)))
    head_height = median_distance(head_origin, marker["HeadTop"])

    parts = [
        AnatomicalPart(
            "pelvis",
            "ellipsoid",
            (
                max(waist_width * 0.50, 0.12 * scale),
                max(waist_depth * 0.52, 0.075 * scale),
                max(root_ab_length * 1.25, 0.070 * scale),
            ),
            TROUSERS_MATERIAL,
            0.5 * (root + abdomen),
            bone_quaternions(bones, root_name),
        ),
        AnatomicalPart(
            "abdomen",
            "ellipsoid",
            (
                max(waist_width * 0.47, 0.12 * scale),
                max(waist_depth * 0.48, 0.070 * scale),
                max(ab_chest_length * 0.62, 0.10 * scale),
            ),
            SHIRT_MATERIAL,
            0.5 * (abdomen + chest),
            bone_quaternions(bones, "Ab"),
        ),
        AnatomicalPart(
            "chest",
            "ellipsoid",
            (
                max(shoulder_width * 0.52, 0.16 * scale),
                max(waist_depth * 0.56, 0.085 * scale),
                max(chest_neck_length * 0.68, 0.14 * scale),
            ),
            SHIRT_MATERIAL,
            0.45 * chest + 0.55 * neck,
            bone_quaternions(bones, "Chest"),
        ),
        AnatomicalPart(
            "neck",
            "capsule",
            (
                0.043 * scale,
                max(neck_head_length * 0.5 - 0.043 * scale, 0.025 * scale),
            ),
            SKIN_MATERIAL,
            0.5 * (neck + head_origin),
            quaternions_aligning_z(neck_head_vector),
        ),
        AnatomicalPart(
            "head",
            "ellipsoid",
            (0.092 * scale, 0.105 * scale, max(head_height * 0.54, 0.115 * scale)),
            SKIN_MATERIAL,
            head_origin + 0.52 * (marker["HeadTop"] - head_origin),
            bone_quaternions(bones, "Head"),
        ),
    ]

    for side in ("L", "R"):
        hand = bones[f"{side}Hand"][:, 4:7]
        finger_base = np.mean(
            np.stack(
                [bones[f"{side}{finger}1"][:, 4:7] for finger in FINGER_NAMES[1:]],
                axis=0,
            ),
            axis=0,
        )
        palm_length = median_distance(hand, finger_base)
        index_base = bones[f"{side}Index1"][:, 4:7]
        pinky_base = bones[f"{side}Pinky1"][:, 4:7]
        palm_width = median_distance(index_base, pinky_base)
        parts.append(
            AnatomicalPart(
                f"{side.lower()}_palm",
                "ellipsoid",
                (
                    max(palm_length * 0.62, 0.050 * scale),
                    max(palm_width * 0.58, 0.030 * scale),
                    0.018 * scale,
                ),
                SKIN_MATERIAL,
                0.5 * (hand + finger_base),
                bone_quaternions(bones, f"{side}Hand"),
            )
        )

    return parts, scale


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
                f'''<body name="marker_{index:02d}_{escape(name)}"
                           mocap="true" pos="{pos}">
                      <geom type="sphere" size="{marker_size:.9g}"
                            rgba="{rgba}" contype="0" conaffinity="0"/>
                    </body>'''
            )

    for index, ((parent, child), midpoint, length) in enumerate(
        zip(edges, segment_midpoints, segment_lengths, strict=True)
    ):
        pos = " ".join(f"{value:.9g}" for value in midpoint)
        nominal_radius, material = segment_style(parent, child, body_scale)
        radius = min(nominal_radius, length * 0.28)
        half_length = max(length * 0.5 - radius, length * 0.12)
        bodies.append(
            f'''<body name="segment_{index:02d}_{escape(parent)}_{escape(child)}"
                       mocap="true" pos="{pos}">
                  <geom type="capsule" size="{radius:.9g} {half_length:.9g}"
                        material="{material}"
                        contype="0" conaffinity="0"/>
                </body>'''
        )

    for part in anatomical_parts:
        pos = " ".join(f"{value:.9g}" for value in part.positions[0])
        size = " ".join(f"{value:.9g}" for value in part.size)
        bodies.append(
            f'''<body name="anatomy_{escape(part.name)}" mocap="true" pos="{pos}">
                  <geom type="{part.geom_type}" size="{size}" material="{part.material}"
                        contype="0" conaffinity="0"/>
                </body>'''
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


def xml_values(values: np.ndarray, precision: int = 7) -> str:
    return " ".join(f"{value:.{precision}g}" for value in values.reshape(-1))


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
            f'''<body name="{body_name}" mocap="true"
                       pos="{xml_values(position)}" quat="{xml_values(quaternion)}"/>'''
        )
        vertex_ids = np.flatnonzero(asset.weights[:, index] > 1e-6)
        vertex_weights = asset.weights[vertex_ids, index]
        skin_bones.append(
            f'''<bone body="{body_name}" bindpos="{xml_values(position)}"
                       bindquat="{xml_values(quaternion)}"
                       vertid="{xml_values(vertex_ids, precision=12)}"
                       vertweight="{xml_values(vertex_weights)}"/>'''
        )

    marker_bodies = []
    if show_markers:
        for index, (name, position) in enumerate(
            zip(marker_names, marker_positions, strict=True)
        ):
            color = marker_color(name, reconstructed).split()
            rgba = " ".join([*color[:3], "0.65"])
            marker_bodies.append(
                f'''<body name="marker_{index:02d}_{escape(name)}" mocap="true"
                           pos="{xml_values(position)}">
                      <geom type="sphere" size="{marker_size:.9g}" rgba="{rgba}"
                            contype="0" conaffinity="0"/>
                    </body>'''
            )

    xml = f'''
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
    '''
    return mujoco.MjModel.from_xml_string(xml)


def visualize(args: argparse.Namespace) -> None:
    marker_motion = load_marker_motion(args.csv)
    markers = np.asarray(marker_motion["positions"])
    marker_names = list(marker_motion["names"])
    reconstructed = set(marker_motion["reconstructed"])
    frame_rate = float(marker_motion["frame_rate"])

    bones = load_bones(args.csv)
    root_name = next(iter(bones))
    scene = None
    if args.scene_obstacles or args.christmas:
        scene = reconstruct_scene_obstacles(
            bones,
            frame_rate,
            args.table_length,
            args.table_width,
            args.wall_width,
            args.wall_height,
            args.christmas,
        )
    scene_xml = "" if scene is None else scene.xml
    cylinder_body_ids = np.empty(0, dtype=np.int32)
    marker_count = len(marker_names) if args.show_markers else 0
    marker_slice = slice(0, marker_count)

    if args.mannequin:
        edges = upper_body_edges()
        segment_pos, segment_quat, segment_lengths = prepare_segments(bones, edges)
        anatomical_parts, body_scale = prepare_anatomical_parts(
            bones, marker_names, markers, root_name
        )
        anatomical_pos = np.stack(
            [part.positions for part in anatomical_parts], axis=1
        )
        anatomical_quat = np.stack(
            [part.quaternions for part in anatomical_parts], axis=1
        )
        model = build_upper_body_model(
            marker_names,
            markers[0],
            reconstructed,
            edges,
            segment_pos[0],
            segment_lengths,
            anatomical_parts,
            body_scale,
            args.marker_size,
            args.show_markers,
            scene_xml,
            args.christmas,
        )
        data = mujoco.MjData(model)
        segment_slice = slice(marker_count, marker_count + len(edges))
        anatomical_slice = slice(
            marker_count + len(edges),
            marker_count + len(edges) + len(anatomical_parts),
        )

        def set_frame(frame: int) -> None:
            if args.show_markers:
                data.mocap_pos[marker_slice] = markers[frame]
                data.mocap_quat[marker_slice] = (1.0, 0.0, 0.0, 0.0)
            data.mocap_pos[segment_slice] = segment_pos[frame]
            data.mocap_quat[segment_slice] = segment_quat[frame]
            data.mocap_pos[anatomical_slice] = anatomical_pos[frame]
            data.mocap_quat[anatomical_slice] = anatomical_quat[frame]
            if scene is not None:
                model.body_pos[cylinder_body_ids] = scene.cylinder_positions[frame]
            data.time = frame / frame_rate
            mujoco.mj_forward(model, data)

        model_description = (
            f"Procedural mannequin: {len(edges)} limb segments, "
            f"{len(anatomical_parts)} anatomical volumes, scale {body_scale:.3f}"
        )
        coverage_description = (
            "torso, head, arms, hands, and articulated fingers (no legs in CSV)"
        )
    else:
        avatar_config = AVATAR_CONFIGS[args.avatar]
        avatar = load_avatar_asset(avatar_config)
        avatar_pos, avatar_quat, avatar_scale = prepare_avatar_pose(
            avatar, bones, marker_names, markers
        )
        model = build_avatar_model(
            avatar_config,
            avatar,
            avatar_scale,
            marker_names,
            markers[0],
            reconstructed,
            args.marker_size,
            args.show_markers,
            scene_xml,
            args.christmas,
        )
        data = mujoco.MjData(model)
        avatar_slice = slice(marker_count, marker_count + len(avatar.joint_names))

        def set_frame(frame: int) -> None:
            if args.show_markers:
                data.mocap_pos[marker_slice] = markers[frame]
                data.mocap_quat[marker_slice] = (1.0, 0.0, 0.0, 0.0)
            data.mocap_pos[avatar_slice] = avatar_pos[frame]
            data.mocap_quat[avatar_slice] = avatar_quat[frame]
            if scene is not None:
                model.body_pos[cylinder_body_ids] = scene.cylinder_positions[frame]
            data.time = frame / frame_rate
            mujoco.mj_forward(model, data)

        model_description = (
            f"{avatar_config.name} upper-body VRM avatar: "
            f"{len(avatar.vertices)} vertices, "
            f"{len(avatar.faces)} faces, {len(avatar.joint_names)} mocap joints, "
            f"scale {avatar_scale:.3f}"
        )
        finger_description = (
            "hands with articulated fingers"
            if any(name.endswith("Proximal") for name in avatar.joint_names)
            else "rigid hands (the VRM has no humanoid finger bones)"
        )
        coverage_description = (
            f"torso, head, arms, and {finger_description}; no lower body"
        )

    if scene is not None:
        cylinder_body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"carried_cylinder_{index}"
                )
                for index in (1, 2)
            ],
            dtype=np.int32,
        )
    set_frame(0)
    print(f"Frames: {len(markers)}, frame rate: {frame_rate:.6g} Hz")
    print(
        f"Markers: {len(marker_names)} "
        f"({'shown' if args.show_markers else 'hidden'})"
    )
    print(model_description)
    print(f"Skeleton root: {root_name}")
    print(f"Body coverage: {coverage_description}")
    if args.christmas:
        print(
            "Christmas surfaces: brick-and-window walls, wooden tables and "
            "shelf, wooden floor with festival carpet, and two animated wine "
            "glasses"
        )
    if scene is not None:
        print(
            "Scene obstacles: two "
            f"{scene.table_length:.3g} x {args.table_width:.3g} x "
            f"{TABLE_HEIGHT:.3g} m table boxes, two perpendicular walls, "
            "one solid floor, and one "
            f"{BOARD_LENGTH:.3g} x {BOARD_WIDTH:.3g} x "
            f"{BOARD_THICKNESS:.3g} m horizontal wall board"
        )
        print(
            "Inferred table top corner XYZ: "
            f"{np.array2string(scene.table_corner, precision=3)}"
        )
        print(
            "Inferred board outer lower-left XYZ: "
            f"{np.array2string(scene.board_corner, precision=3)} "
            f"(measured hand Z {scene.measured_board_corner_height:.3f} m)"
        )
        print(
            "Wall/table corner alignment XY: "
            f"{np.array2string(scene.wall_contact[:2], precision=3)}; "
            f"extended wall span {scene.wall_span:.3f} m; "
            f"clearance {TABLE_WALL_CLEARANCE * 1000:.0f} mm"
        )
        pickup_times = [frame / frame_rate for frame in scene.cylinder_pickup_frames]
        release_times = [frame / frame_rate for frame in scene.cylinder_release_frames]
        print(
            "Cylinder transports (pickup -> release): "
            + ", ".join(
                f"{pickup:.2f}s -> {release:.2f}s"
                for pickup, release in zip(
                    pickup_times, release_times, strict=True
                )
            )
        )
    print(
        "Reconstructed marker: "
        + (", ".join(sorted(reconstructed)) if reconstructed else "none")
    )

    if args.check_only:
        print(f"MuJoCo mocap bodies: {model.nmocap}; model validation OK")
        return

    from mujoco import viewer as mjviewer

    state = {"paused": False, "reset": False}

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):
            state["reset"] = True

    frame = 0
    fractional_frames = 0.0
    previous_time = time.perf_counter()
    center = markers.mean(axis=(0, 1))

    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = center
        motion_extent = np.ptp(markers.reshape(-1, 3), axis=0).max()
        if scene is not None:
            # View from the open north-west side of the L-shaped room. The
            # previous south-east view looked through both walls and obscured
            # the avatar, tables, and shelf.
            viewer.cam.distance = max(4.5, motion_extent * 1.25)
            viewer.cam.azimuth = 315
            viewer.cam.elevation = -24
        else:
            viewer.cam.distance = max(3.0, motion_extent * 1.6)
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -18
        print("Controls: Space = pause/resume, R = restart, Esc = close")

        while viewer.is_running():
            now = time.perf_counter()
            elapsed = min(now - previous_time, 0.1)
            previous_time = now
            if state["reset"]:
                frame = 0
                fractional_frames = 0.0
                state["reset"] = False
            if not state["paused"]:
                fractional_frames += elapsed * frame_rate * args.speed
                advance = int(fractional_frames)
                fractional_frames -= advance
                if advance:
                    frame += advance
                    if frame >= len(markers):
                        if args.no_loop:
                            frame = len(markers) - 1
                            state["paused"] = True
                        else:
                            frame %= len(markers)
            set_frame(frame)
            viewer.sync()
            time.sleep(0.001)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help="OptiTrack CSV to visualize (default: %(default)s)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    marker_display = parser.add_mutually_exclusive_group()
    marker_display.add_argument(
        "--show-markers",
        dest="show_markers",
        action="store_true",
        help="overlay the tracked marker spheres on the avatar",
    )
    marker_display.add_argument(
        "--avatar-only",
        "--hide-markers",
        dest="show_markers",
        action="store_false",
        help="hide tracked marker points and show the avatar without their overlay",
    )
    parser.set_defaults(show_markers=False)
    parser.add_argument(
        "--avatar",
        choices=sorted(AVATAR_CONFIGS),
        default=DEFAULT_AVATAR,
        help="VRM avatar to use (default: %(default)s)",
    )
    parser.add_argument(
        "--mannequin",
        action="store_true",
        help="use the procedural mannequin instead of the selected VRM avatar",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.011,
        help="marker radius in meters when --show-markers is used",
    )
    parser.add_argument(
        "--scene-obstacles",
        action="store_true",
        help=(
            "reconstruct two table boxes and the wall-mounted board from the "
            "start/end calibration poses"
        ),
    )
    parser.add_argument(
        "--christmas",
        action="store_true",
        help=(
            "use Christmas window/wood/carpet surfaces and wine glasses; "
            "implies --scene-obstacles"
        ),
    )
    parser.add_argument(
        "--table-length",
        type=float,
        default=None,
        help=(
            "override the auto-fitted length of each end-to-end table box "
            "in meters"
        ),
    )
    parser.add_argument(
        "--table-width",
        type=float,
        default=0.65,
        help="width of each table box in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--wall-width",
        type=float,
        default=4.0,
        help="wall width in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--wall-height",
        type=float,
        default=2.6,
        help="wall height in meters (default: %(default)s)",
    )
    parser.add_argument("--no-loop", action="store_true", help="stop on the last frame")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate data/model without opening a GUI",
    )
    return parser


if __name__ == "__main__":
    visualize(make_parser().parse_args())
