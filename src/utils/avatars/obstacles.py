"""Reusable avatars obstacles helpers."""

from __future__ import annotations

import numpy as np

from utils.avatars.models import (
    SceneObstacles,
)
from utils.avatars.pose import (
    matrices_to_mujoco_quaternions,
)
from utils.avatars.scene_xml import (
    static_box_xml,
    xml_values,
)

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


def horizontal_body_axes(
    bones: dict[str, np.ndarray], frame_slice: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal forward and left axes from a stable pose window."""
    left = np.median(
        bones["LShoulder"][frame_slice, 4:7] - bones["RShoulder"][frame_slice, 4:7],
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
            1.0 - alpha[:, None]
        ) * initial_position + alpha[:, None] * grip_positions[pickup_frame:pickup_end]
    track[pickup_end:release_start] = grip_positions[pickup_end:release_start]
    if release_frame > release_start:
        alpha = np.linspace(0.0, 1.0, release_frame - release_start, endpoint=False)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        track[release_start:release_frame] = (1.0 - alpha[:, None]) * grip_positions[
            release_start:release_frame
        ] + alpha[:, None] * final_position
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
    table_forward, table_left = snap_horizontal_axes_to_grid(measured_table_forward)
    measured_wall_normal, _ = horizontal_body_axes(bones, end_slice)
    wall_normal, wall_left = snap_horizontal_axes_to_grid(measured_wall_normal)

    table_corner = np.median(bones["RHand"][start_slice, 4:7], axis=0)
    table_corner[2] = TABLE_HEIGHT
    root_name = next(iter(bones))
    avatar_start = np.median(bones[root_name][start_slice, 4:7], axis=0)
    table_width_direction = table_left.copy()
    if np.dot(table_width_direction, table_corner - avatar_start) < 0.0:
        table_width_direction *= -1.0
    measured_board_corner = np.median(bones["LHand"][end_slice, 4:7], axis=0)
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
        normal_distance = float(np.dot(wall_surface - table_corner, wall_normal))
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
    floor_span_along_wall = abs(opposite_offset - return_wall_offset) + WALL_THICKNESS
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
        'material="christmas_floor"' if christmas else 'rgba="0.32 0.32 0.32 1"'
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
            np.array([BOARD_LENGTH * 0.5, BOARD_WIDTH * 0.5, BOARD_THICKNESS * 0.5]),
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
            [bones[name][:, 4:7] for name in ("RThumb3", "RIndex3", "RMiddle3")],
            axis=0,
        ),
        axis=0,
    )
    frame_count = len(grip_positions)
    pickup_frames = tuple(
        min(frame_count - 1, round(seconds * frame_rate)) for seconds in (2.58, 16.92)
    )
    release_frames = tuple(
        min(frame_count - 1, round(seconds * frame_rate)) for seconds in (11.58, 23.08)
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
        indicated = np.median(grip_positions[sample_start:release_frame], axis=0)
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
                "</body>"
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
