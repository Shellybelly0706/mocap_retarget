#!/usr/bin/env python3
"""Visualize a MoSh++ SMPL-X Stage-II result as a skinned MuJoCo body.

The viewer uses MuJoCo's native ``<skin>`` renderer.  Shape blend shapes are
baked into the bind mesh, while the 55 SMPL-X joints drive linear-blend
skinning at runtime.  The body, table, and weightless cup are kinematic
visualization objects.  The cup stays on the table through the pickup frame,
then follows the right hand as it retracts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "moshpp_outputs/Zekun_upper_righthand_added_1_fingers_full/1112/c3d"
    / "Skeleton_Zixuan/Zekun_upper_righthand_added_1_stageii.npz"
)
DEFAULT_MODEL = ROOT / "mocap_data/data/smplx/SMPLX_NEUTRAL.npz"

# SMPL-X is Y-up.  MuJoCo scenes conventionally use Z-up.  For row vectors,
# points are converted with p_mj = p_smplx @ SMPLX_TO_MUJOCO.T.
SMPLX_TO_MUJOCO = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

REQUIRED_RESULT_KEYS = ("trans", "root_orient", "pose_body", "pose_hand", "betas")
REQUIRED_MODEL_KEYS = (
    "v_template",
    "f",
    "weights",
    "J_regressor",
    "kintree_table",
    "shapedirs",
)

FINGER_LABEL_FALLBACKS = {
    "RIndex": "CustomMarker6",
    "RMiddle": "CustomMarker4",
    "RHandIn": "CustomMarker1",
}


@dataclass(frozen=True)
class StaticScene:
    table_center_xy: np.ndarray
    table_axis_x: np.ndarray
    table_axis_y: np.ndarray
    table_top_z: float
    table_length: float
    table_width: float
    table_thickness: float
    cup_center: np.ndarray
    cup_positions: np.ndarray
    cup_quaternions: np.ndarray
    cup_height: float
    cup_radius: float
    index_contact: np.ndarray
    middle_contact: np.ndarray
    contact_frame: int
    pickup_frame: int
    grip_frame: int
    cup_is_static: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a MoSh++ SMPL-X Stage-II NPZ as a skinned MuJoCo body."
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--start-frame", type=int, default=60)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--table-length", type=float, default=1.05)
    parser.add_argument("--table-width", type=float, default=0.75)
    parser.add_argument("--table-thickness", type=float, default=0.045)
    parser.add_argument("--cup-height", type=float, default=0.16)
    parser.add_argument("--cup-radius", type=float, default=0.040)
    parser.add_argument(
        "--contact-frame",
        type=int,
        default=60,
        help="Frame at which the right index/middle fingers touch the table corner.",
    )
    parser.add_argument(
        "--pickup-frame",
        type=int,
        default=130,
        help="Earliest frame from which hand contact may initiate the pickup.",
    )
    parser.add_argument(
        "--grip-transition-frames",
        type=int,
        default=20,
        help="Frames used to move the cup from the tabletop into full-hand contact.",
    )
    parser.add_argument(
        "--keep-cup-static",
        action="store_true",
        help="Keep the cup on the table for the complete animation.",
    )
    parser.add_argument(
        "--no-loop", action="store_true", help="Pause on the final frame instead of looping."
    )
    parser.add_argument(
        "--keep-world-position",
        action="store_true",
        help="Do not horizontally center the initial pelvis at the world origin.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load data and compile the MuJoCo skin without opening a GUI.",
    )
    return parser.parse_args()


def require_keys(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise KeyError(f"{name} is missing required arrays: {', '.join(missing)}")


def axis_angle_to_matrix(rotvecs: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion for arrays ending in three values."""
    vectors = np.asarray(rotvecs, dtype=np.float64)
    original_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, 3)
    angles = np.linalg.norm(flat, axis=1)
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None], len(flat), axis=0)

    active = angles > 1e-12
    if np.any(active):
        axes = flat[active] / angles[active, None]
        x, y, z = axes.T
        zeros = np.zeros_like(x)
        skew = np.stack(
            (zeros, -z, y, z, zeros, -x, -y, x, zeros), axis=1
        ).reshape(-1, 3, 3)
        sine = np.sin(angles[active])[:, None, None]
        cosine = np.cos(angles[active])[:, None, None]
        identity = np.eye(3, dtype=np.float64)[None]
        matrices[active] = identity + sine * skew + (1.0 - cosine) * (skew @ skew)

    return matrices.reshape(*original_shape, 3, 3)


def matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to MuJoCo quaternions in w, x, y, z order."""
    matrices = np.asarray(matrices, dtype=np.float64)
    quaternions = np.empty((*matrices.shape[:-2], 4), dtype=np.float64)
    flat_matrices = matrices.reshape(-1, 3, 3)
    flat_quaternions = quaternions.reshape(-1, 4)
    for matrix, quaternion in zip(flat_matrices, flat_quaternions, strict=True):
        mujoco.mju_mat2Quat(quaternion, matrix.reshape(9))
    return quaternions


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def skinned_vertices_at_frame(
    vertices: np.ndarray,
    bind_positions: np.ndarray,
    weights: np.ndarray,
    joint_positions: np.ndarray,
    joint_quaternions: np.ndarray,
    frame: int,
) -> np.ndarray:
    """Evaluate the same linear-blend skin used by MuJoCo for one frame."""
    output = np.zeros_like(vertices, dtype=np.float64)
    for joint in range(len(bind_positions)):
        active = weights[:, joint] > 1e-8
        if not np.any(active):
            continue
        rotation = quaternion_to_matrix(joint_quaternions[frame, joint])
        transformed = (
            (vertices[active] - bind_positions[joint]) @ rotation.T
            + joint_positions[frame, joint]
        )
        output[active] += weights[active, joint, None] * transformed
    return output


def hand_marker_vertex_ids(result: np.lib.npyio.NpzFile) -> tuple[int, int, int]:
    marker_meta = result["marker_meta"].item()
    marker_vids = dict(marker_meta["marker_vids"])
    equivalent_labels: dict[str, str] = {}
    layout_path = Path(str(marker_meta.get("marker_layout_fname", "")))
    if layout_path.is_file():
        with layout_path.open("r", encoding="utf-8") as handle:
            layout = json.load(handle)
        equivalent_labels = dict(
            layout.get("layout_metadata", {}).get("equivalent_label_map", {})
        )

    def resolve(target: str) -> int:
        if target in marker_vids:
            return int(marker_vids[target])
        for source, equivalent in equivalent_labels.items():
            if equivalent == target and source in marker_vids:
                return int(marker_vids[source])
        fallback = FINGER_LABEL_FALLBACKS[target]
        if fallback in marker_vids:
            return int(marker_vids[fallback])
        raise KeyError(f"Could not resolve {target} in Stage-II marker metadata")

    return resolve("RIndex"), resolve("RMiddle"), resolve("RHandIn")


def finger_marker_vertex_ids(result: np.lib.npyio.NpzFile) -> tuple[int, int]:
    index, middle, _ = hand_marker_vertex_ids(result)
    return index, middle


def build_static_scene(
    result: np.lib.npyio.NpzFile,
    vertices: np.ndarray,
    faces: np.ndarray,
    bind_positions: np.ndarray,
    weights: np.ndarray,
    joint_positions: np.ndarray,
    joint_quaternions: np.ndarray,
    table_length: float,
    table_width: float,
    table_thickness: float,
    cup_height: float,
    cup_radius: float,
    contact_frame: int,
    pickup_frame: int,
    grip_transition_frames: int,
    animate_cup_pickup: bool,
) -> StaticScene:
    skin_frames = [
        skinned_vertices_at_frame(
            vertices,
            bind_positions,
            weights,
            joint_positions,
            joint_quaternions,
            frame,
        )
        for frame in range(len(joint_positions))
    ]
    contact_vertices = skin_frames[contact_frame]
    index_vertex, middle_vertex, palm_marker_vertex = hand_marker_vertex_ids(result)
    index_contact = contact_vertices[index_vertex]
    middle_contact = contact_vertices[middle_vertex]
    contact_midpoint = 0.5 * (index_contact + middle_contact)

    root_xy = joint_positions[contact_frame, 0, :2]
    outward = contact_midpoint[:2] - root_xy
    outward_norm = np.linalg.norm(outward)
    if outward_norm < 1e-8:
        outward = np.array([0.0, 1.0])
    else:
        outward /= outward_norm

    half_length = 0.5 * table_length
    half_width = 0.5 * table_width
    diagonal_angle = np.arctan2(half_width, half_length)
    cosine = np.cos(-diagonal_angle)
    sine = np.sin(-diagonal_angle)
    axis_x = np.array(
        [cosine * outward[0] - sine * outward[1],
         sine * outward[0] + cosine * outward[1]]
    )
    axis_y = np.array([-axis_x[1], axis_x[0]])

    # Marker vertices are not necessarily the outermost fingertip vertices.
    # This marker-to-corner offset puts the actual skin surface at the corner.
    corner_clearance = 0.012
    vertical_clearance = 0.001
    near_corner = contact_midpoint[:2] + corner_clearance * outward
    table_center_xy = (
        near_corner + half_length * axis_x + half_width * axis_y
    )
    table_top_z = float(min(index_contact[2], middle_contact[2]) - vertical_clearance)

    # The cup starts just beyond the fingertips.  The extra 35 mm clears both
    # its solid volume and the slightly wider decorative rim.
    cup_xy = contact_midpoint[:2] + (cup_radius + 0.035) * outward
    cup_center = np.array(
        [cup_xy[0], cup_xy[1], table_top_z + 0.002 + 0.5 * cup_height]
    )

    # Fit the moving cup to the palm side of the hand, rather than attaching it
    # to a fingertip or to one fixed wrist offset.  Candidate cylinder centres
    # must remain outside the complete hand surface, touch the palm region, and
    # preferably touch additional fingers.  A short dynamic-programming pass
    # chooses the smoothest candidate sequence over the remaining motion.
    rim_radius = max(0.003, min(0.006, cup_radius * 0.09))
    cup_half_height = 0.5 * cup_height
    rim_center_height = cup_half_height + 0.002
    cup_footprint_radius = cup_radius + rim_radius

    def cup_intersects_skin(skin: np.ndarray, centre: np.ndarray) -> bool:
        delta = skin - centre
        radial_distance = np.linalg.norm(delta[:, :2], axis=1)
        in_body = (
            (radial_distance < cup_radius)
            & (np.abs(delta[:, 2]) < cup_half_height)
        )
        # The visible rim is a ring of capsules, not a wider cylinder along the
        # full cup height. Treating it separately lets the hand touch the body.
        rim_distance = np.sqrt(
            (radial_distance - cup_radius) ** 2
            + (delta[:, 2] - rim_center_height) ** 2
        )
        in_rim = rim_distance < rim_radius
        in_opening = (
            (radial_distance < 0.82 * cup_radius)
            & (np.abs(delta[:, 2] - (cup_half_height + 0.0025)) < 0.0015)
        )
        return bool(np.any(in_body | in_rim | in_opening))
    grip_frame = min(
        pickup_frame + grip_transition_frames,
        len(joint_positions) - 1,
    )
    dominant_joint = np.argmax(weights, axis=1)
    hand_contact_region = (dominant_joint == 21) | (
        (dominant_joint >= 40) & (dominant_joint <= 54)
    )

    def hand_clearance(skin: np.ndarray, centre: np.ndarray) -> float:
        delta = skin - centre
        active = hand_contact_region & (
            np.abs(delta[:, 2]) < cup_half_height
        )
        return float(
            np.linalg.norm(delta[active, :2], axis=1).min() - cup_radius
        )

    def table_pose_toward_cup(
        cup_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search a corner-anchored pose containing the cup without body overlap."""
        slab_points: list[np.ndarray] = []
        for skin in skin_frames:
            in_slab_height = (
                (skin[:, 2] < table_top_z)
                & (skin[:, 2] > table_top_z - table_thickness)
            )
            shallow_hand_contact = (
                hand_contact_region
                & (skin[:, 2] >= table_top_z - 0.008)
            )
            slab_points.append(skin[in_slab_height & ~shallow_hand_contact, :2])
        forbidden_points = np.concatenate(slab_points, axis=0)

        poses: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        original_axis_x = axis_x.copy()
        for angle_degrees in range(0, 360, 2):
            angle = np.deg2rad(angle_degrees)
            candidate_axis_x = np.array([np.cos(angle), np.sin(angle)])
            candidate_axis_y = np.array(
                [-candidate_axis_x[1], candidate_axis_x[0]]
            )
            for corner_x in (-1.0, 1.0):
                for corner_y in (-1.0, 1.0):
                    candidate_center = (
                        near_corner
                        - corner_x * half_length * candidate_axis_x
                        - corner_y * half_width * candidate_axis_y
                    )
                    cup_relative = cup_xy - candidate_center
                    if (
                        abs(cup_relative @ candidate_axis_x)
                        + cup_footprint_radius
                        > half_length
                        or abs(cup_relative @ candidate_axis_y)
                        + cup_footprint_radius
                        > half_width
                    ):
                        continue
                    relative = forbidden_points - candidate_center
                    inside = (
                        (np.abs(relative @ candidate_axis_x) < half_length)
                        & (np.abs(relative @ candidate_axis_y) < half_width)
                    )
                    if np.any(inside):
                        continue
                    orientation_cost = 1.0 - abs(
                        float(candidate_axis_x @ original_axis_x)
                    )
                    poses.append(
                        (
                            orientation_cost,
                            candidate_center,
                            candidate_axis_x,
                            candidate_axis_y,
                        )
                    )
        if not poses:
            raise RuntimeError(
                "Could not orient the enlarged tabletop around the touched "
                "corner without intersecting the body"
            )
        _, table_center, table_axis_x, table_axis_y = min(
            poses, key=lambda pose: pose[0]
        )
        return table_center, table_axis_x, table_axis_y

    if not animate_cup_pickup:
        # The fitted hand stays open, so a pickup would necessarily be an
        # artificial cup motion. Keep the cup fixed and move its tabletop
        # position only enough to clear the complete swept hand/forearm volume.
        static_collision_region = (dominant_joint == 19) | hand_contact_region
        swept_hand = np.concatenate(
            [skin[static_collision_region] for skin in skin_frames], axis=0
        )
        static_candidates: list[tuple[float, np.ndarray]] = []
        for offset in np.arange(0.0, 0.201, 0.002):
            for angle_degrees in range(0, 360, 5):
                angle = np.deg2rad(angle_degrees)
                candidate = cup_center.copy()
                candidate[:2] += offset * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                if cup_intersects_skin(swept_hand, candidate):
                    continue
                delta = swept_hand - candidate
                active = np.abs(delta[:, 2]) < cup_half_height
                clearance = float(
                    np.linalg.norm(delta[active, :2], axis=1).min()
                    - cup_radius
                )
                static_candidates.append((clearance, candidate))
            if static_candidates:
                break
        if not static_candidates:
            raise RuntimeError(
                "Could not place a stationary cup outside the swept hand volume"
            )

        cup_center = min(static_candidates, key=lambda item: item[0])[1]
        table_center_xy, axis_x, axis_y = table_pose_toward_cup(cup_center[:2])
        cup_positions = np.repeat(
            cup_center[None, :], len(joint_positions), axis=0
        )
        cup_quaternions = np.zeros((len(joint_positions), 4), dtype=np.float64)
        cup_quaternions[:, 0] = 1.0

        leg_inset = min(0.07, 0.22 * min(table_length, table_width))
        leg_half_width = 0.032
        leg_centres = (
            (-half_length + leg_inset, -half_width + leg_inset),
            (-half_length + leg_inset, half_width - leg_inset),
            (half_length - leg_inset, -half_width + leg_inset),
            (half_length - leg_inset, half_width - leg_inset),
        )
        for frame, skin in enumerate(skin_frames):
            relative = skin[:, :2] - table_center_xy
            local_x = relative @ axis_x
            local_y = relative @ axis_y
            in_tabletop = (
                (np.abs(local_x) < half_length)
                & (np.abs(local_y) < half_width)
                & (skin[:, 2] < table_top_z)
                & (skin[:, 2] > table_top_z - table_thickness)
            )
            in_leg = np.zeros(len(skin), dtype=bool)
            for leg_x, leg_y in leg_centres:
                in_leg |= (
                    (np.abs(local_x - leg_x) < leg_half_width)
                    & (np.abs(local_y - leg_y) < leg_half_width)
                    & (skin[:, 2] > 0.0)
                    & (skin[:, 2] < table_top_z - table_thickness)
                )
            if np.any(in_tabletop | in_leg):
                raise RuntimeError(
                    f"Table intersects the SMPL-X skin at frame {frame}"
                )
            if cup_intersects_skin(skin, cup_center):
                raise RuntimeError(
                    f"Stationary cup intersects the SMPL-X skin at frame {frame}"
                )

        return StaticScene(
            table_center_xy=table_center_xy,
            table_axis_x=axis_x,
            table_axis_y=axis_y,
            table_top_z=table_top_z,
            table_length=table_length,
            table_width=table_width,
            table_thickness=table_thickness,
            cup_center=cup_center,
            cup_positions=cup_positions,
            cup_quaternions=cup_quaternions,
            cup_height=cup_height,
            cup_radius=cup_radius,
            index_contact=index_contact,
            middle_contact=middle_contact,
            contact_frame=contact_frame,
            pickup_frame=-1,
            grip_frame=-1,
            cup_is_static=True,
        )

    finger_joint_groups = (
        (40, 41, 42),   # index
        (43, 44, 45),   # middle
        (46, 47, 48),   # pinky
        (49, 50, 51),   # ring
        (52, 53, 54),   # thumb
    )

    # The requested contact is on the surface across the hand thickness from
    # ``RHandIn``. Find the closest right-hand vertex whose surface normal is
    # opposed to the marker normal, then grow a connected patch on that face.
    adjacency = [set() for _ in range(len(vertices))]
    for triangle in faces:
        first, second, third = (int(vertex) for vertex in triangle)
        adjacency[first].update((second, third))
        adjacency[second].update((first, third))
        adjacency[third].update((first, second))
    bind_normals = np.zeros_like(vertices)
    bind_triangles = vertices[faces]
    bind_face_normals = np.cross(
        bind_triangles[:, 1] - bind_triangles[:, 0],
        bind_triangles[:, 2] - bind_triangles[:, 0],
    )
    np.add.at(bind_normals, faces[:, 0], bind_face_normals)
    np.add.at(bind_normals, faces[:, 1], bind_face_normals)
    np.add.at(bind_normals, faces[:, 2], bind_face_normals)
    bind_normals /= np.linalg.norm(bind_normals, axis=1, keepdims=True)
    marker_distances = np.linalg.norm(
        vertices - vertices[palm_marker_vertex], axis=1
    )
    normal_alignment = bind_normals @ bind_normals[palm_marker_vertex]
    opposite_candidates = np.flatnonzero(
        (dominant_joint == 21)
        & (marker_distances < 0.06)
        & (normal_alignment < -0.8)
    )
    if not len(opposite_candidates):
        raise RuntimeError("Could not resolve the palm surface opposite RHandIn")
    palm_vertex = int(
        opposite_candidates[np.argmin(marker_distances[opposite_candidates])]
    )
    palm_patch = {palm_vertex}
    frontier = {palm_vertex}
    for _ in range(6):
        frontier = set().union(*(adjacency[vertex] for vertex in frontier))
        frontier -= palm_patch
        palm_patch |= frontier
    palm_patch_vertices = np.asarray(sorted(palm_patch), dtype=np.int32)
    palmar_alignment = bind_normals[palm_patch_vertices] @ bind_normals[palm_vertex]
    palm_patch_vertices = palm_patch_vertices[
        palmar_alignment >= np.cos(np.deg2rad(70.0))
    ]
    palm_incident_faces = faces[np.any(faces == palm_vertex, axis=1)]

    def inner_palm_direction(skin: np.ndarray) -> np.ndarray:
        triangles = skin[palm_incident_faces]
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ).sum(axis=0)
        normal_xy = normal[:2]
        normal_xy_norm = np.linalg.norm(normal_xy)
        if normal_xy_norm < 1e-8:
            raise RuntimeError("Opposite-palm surface normal has no horizontal component")
        return normal_xy / normal_xy_norm

    palm_directions = [inner_palm_direction(skin) for skin in skin_frames]
    # At the pickup frame the cup must be on the inner-palm side. Later poses do
    # not always form a closed grasp, so the solver keeps any collision-free
    # right-hand contact and rewards palm/index/thumb contact instead of forcing
    # an impossible penetration.
    minimum_inner_palm_alignment = float(np.cos(np.deg2rad(80.0)))
    candidate_frames: list[list[tuple[np.ndarray, int]]] = []
    radial_samples = np.arange(
        cup_radius + 0.0005,
        0.171,
        0.001,
    )
    contact_gap = 0.0025
    tabletop_cup_z = table_top_z + cup_half_height
    vertical_grip_offset = (
        tabletop_cup_z - skin_frames[grip_frame][palm_vertex, 2]
    )
    for frame in range(pickup_frame + 1, len(joint_positions)):
        skin = skin_frames[frame]
        palm = skin[palm_vertex]
        candidate_z = palm[2] + vertical_grip_offset
        nearby = (
            (np.abs(skin[:, 2] - candidate_z) < cup_half_height)
            & (np.linalg.norm(skin[:, :2] - palm[:2], axis=1) < 0.25)
        )
        nearby_points = skin[nearby, :2]
        nearby_joints = dominant_joint[nearby]
        palm_patch_points = skin[palm_patch_vertices]
        palm_patch_points = palm_patch_points[
            np.abs(palm_patch_points[:, 2] - candidate_z)
            < cup_half_height,
            :2,
        ]
        palm_direction = palm_directions[frame]
        frame_candidates: list[tuple[np.ndarray, int]] = []
        angle_step = 2 if frame == grip_frame else 10
        for angle_degrees in range(0, 360, angle_step):
            angle = np.deg2rad(angle_degrees)
            direction = np.array([np.cos(angle), np.sin(angle)])
            if (
                frame == grip_frame
                and direction @ palm_direction < minimum_inner_palm_alignment
            ):
                continue
            centres = palm[:2] + radial_samples[:, None] * direction
            clearances = np.linalg.norm(
                nearby_points[None, :, :] - centres[:, None, :],
                axis=2,
            ).min(axis=1)
            safe = np.flatnonzero(clearances >= cup_radius)
            if not len(safe):
                continue
            sample = next(
                (
                    int(candidate)
                    for candidate in safe
                    if not cup_intersects_skin(
                        skin,
                        np.array(
                            [
                                centres[candidate, 0],
                                centres[candidate, 1],
                                candidate_z,
                            ]
                        ),
                    )
                ),
                None,
            )
            if sample is None:
                continue
            centre_xy = centres[sample]
            distances = np.linalg.norm(nearby_points - centre_xy, axis=1)
            palm_gap = (
                float(np.linalg.norm(palm_patch_points - centre_xy, axis=1).min())
                - cup_radius
                if len(palm_patch_points)
                else np.inf
            )
            group_clearances: list[float] = []
            for group in finger_joint_groups:
                group_mask = np.isin(nearby_joints, group)
                group_clearances.append(
                    float(distances[group_mask].min())
                    if np.any(group_mask)
                    else np.inf
                )
            group_gaps = [clearance - cup_radius for clearance in group_clearances]
            index_contacting = group_gaps[0] <= contact_gap
            thumb_contacting = group_gaps[4] <= contact_gap
            palm_contacting = palm_gap <= contact_gap
            hand_gap = float(distances.min() - cup_radius)
            if hand_gap > contact_gap:
                continue
            if frame == grip_frame and not (
                palm_contacting and index_contacting and thumb_contacting
            ):
                continue
            primary_contacts = sum(
                (palm_contacting, index_contacting, thumb_contacting)
            )
            other_finger_contacts = sum(
                gap <= contact_gap for gap in group_gaps[1:4]
            )
            contact_count = 3 * primary_contacts + other_finger_contacts
            frame_candidates.append(
                (np.array([centre_xy[0], centre_xy[1], candidate_z]), contact_count)
            )
        if not frame_candidates:
            raise RuntimeError(
                f"Could not find a collision-free right-hand contact at frame {frame}"
            )
        candidate_frames.append(frame_candidates)

    contact_reward = 0.0005
    costs = -contact_reward * np.array(
        [candidate[1] for candidate in candidate_frames[0]],
        dtype=np.float64,
    )
    back_pointers: list[np.ndarray] = []
    for offset in range(1, len(candidate_frames)):
        frame = pickup_frame + 1 + offset
        previous_centres = np.array(
            [candidate[0] for candidate in candidate_frames[offset - 1]]
        )
        current_centres = np.array(
            [candidate[0] for candidate in candidate_frames[offset]]
        )
        previous_palm = skin_frames[frame - 1][palm_vertex]
        current_palm = skin_frames[frame][palm_vertex]
        previous_offsets = previous_centres - previous_palm
        current_offsets = current_centres - current_palm
        transition_cost = np.linalg.norm(
            current_offsets[:, None, :] - previous_offsets[None, :, :],
            axis=2,
        ) ** 2
        total_cost = transition_cost + costs[None, :]
        back_pointer = np.argmin(total_cost, axis=1)
        contact_counts = np.array(
            [candidate[1] for candidate in candidate_frames[offset]],
            dtype=np.float64,
        )
        costs = (
            total_cost[np.arange(len(current_centres)), back_pointer]
            - contact_reward * contact_counts
        )
        back_pointers.append(back_pointer)

    selected = int(np.argmin(costs))
    selected_candidates = [selected]
    for back_pointer in reversed(back_pointers):
        selected = int(back_pointer[selected])
        selected_candidates.append(selected)
    selected_candidates.reverse()
    contact_path = np.array(
        [
            candidate_frames[offset][candidate][0]
            for offset, candidate in enumerate(selected_candidates)
        ]
    )

    grip_path_index = grip_frame - pickup_frame - 1
    moving_path = contact_path[grip_path_index:].copy()

    # Put the stationary cup at the horizontal position where the hand reaches
    # it on the grip frame. The hand approaches the cup; the cup never travels
    # toward an open hand.
    cup_center = moving_path[0].copy()
    cup_center[2] = tabletop_cup_z
    cup_positions = np.repeat(cup_center[None, :], len(joint_positions), axis=0)
    cup_positions[grip_frame:] = moving_path
    contact_start_frame = grip_frame
    table_center_xy, axis_x, axis_y = table_pose_toward_cup(cup_center[:2])

    first_cup_local = cup_center[:2] - table_center_xy
    first_cup_local_x = abs(first_cup_local @ axis_x)
    first_cup_local_y = abs(first_cup_local @ axis_y)
    if (
        first_cup_local_x + cup_footprint_radius > half_length
        or first_cup_local_y + cup_footprint_radius > half_width
    ):
        raise RuntimeError(
            "The complete cup footprint is not on the tabletop at frame 0"
        )

    # Keep the solid bottom above the tabletop until the cup has cleared its
    # footprint.  This correction is normally active only at the start of lift.
    for frame in range(contact_start_frame + 1, len(joint_positions)):
        cup_relative = cup_positions[frame, :2] - table_center_xy
        still_over_table = (
            abs(cup_relative @ axis_x) < half_length + cup_footprint_radius
            and abs(cup_relative @ axis_y) < half_width + cup_footprint_radius
        )
        if still_over_table:
            cup_positions[frame, 2] = max(
                cup_positions[frame, 2],
                table_top_z + 0.5 * cup_height,
            )

    cup_quaternions = np.zeros((len(joint_positions), 4), dtype=np.float64)
    cup_quaternions[:, 0] = 1.0

    # MuJoCo's skin is visual-only, so enforce the requested no-penetration
    # condition explicitly against every frame, including the visible cup rim.
    leg_inset = min(0.07, 0.22 * min(table_length, table_width))
    leg_half_width = 0.032
    leg_centres = (
        (-half_length + leg_inset, -half_width + leg_inset),
        (-half_length + leg_inset, half_width - leg_inset),
        (half_length - leg_inset, -half_width + leg_inset),
        (half_length - leg_inset, half_width - leg_inset),
    )
    for frame in range(len(joint_positions)):
        skin = skin_frames[frame]
        relative = skin[:, :2] - table_center_xy
        local_x = relative @ axis_x
        local_y = relative @ axis_y
        in_tabletop = (
            (np.abs(local_x) < half_length)
            & (np.abs(local_y) < half_width)
            & (skin[:, 2] < table_top_z)
            & (skin[:, 2] > table_top_z - table_thickness)
        )
        in_leg = np.zeros(len(skin), dtype=bool)
        for leg_x, leg_y in leg_centres:
            in_leg |= (
                (np.abs(local_x - leg_x) < leg_half_width)
                & (np.abs(local_y - leg_y) < leg_half_width)
                & (skin[:, 2] > 0.0)
                & (skin[:, 2] < table_top_z - table_thickness)
            )
        # The reaching fingers are meant to touch the tabletop corner. Permit
        # only a small skin/render tolerance there; all non-hand and leg
        # intersections remain invalid.
        shallow_hand_contact = (
            in_tabletop
            & hand_contact_region
            & (skin[:, 2] >= table_top_z - 0.008)
        )
        invalid_table_collision = (in_tabletop | in_leg) & ~shallow_hand_contact
        if np.any(invalid_table_collision):
            collision_type = "top" if np.any(in_tabletop & ~shallow_hand_contact) else "leg"
            collision_ids = np.flatnonzero(invalid_table_collision)
            maximum_depth = float(
                np.max(table_top_z - skin[collision_ids, 2])
            )
            collision_joints = sorted(
                set(int(joint) for joint in dominant_joint[collision_ids])
            )
            raise RuntimeError(
                f"Table {collision_type} intersects the SMPL-X skin at frame "
                f"{frame} ({len(collision_ids)} vertices; first={collision_ids[0]}; "
                f"depth={maximum_depth:.4f} m; joints={collision_joints})"
            )
        if cup_intersects_skin(skin, cup_positions[frame]):
            raise RuntimeError(f"Cup intersects the SMPL-X skin at frame {frame}")

        if frame >= grip_frame:
            cup_delta_xy = np.linalg.norm(
                skin[:, :2] - cup_positions[frame, :2],
                axis=1,
            )
            active_height = (
                np.abs(skin[:, 2] - cup_positions[frame, 2])
                < cup_half_height
            )
            right_hand_clearance = hand_clearance(skin, cup_positions[frame])
            if right_hand_clearance > contact_gap:
                raise RuntimeError(
                    f"Cup loses right-hand contact at frame {frame}: "
                    f"clearance={right_hand_clearance:.4f} m"
                )

            if frame == grip_frame:
                palm_region = np.zeros(len(skin), dtype=bool)
                palm_region[palm_patch_vertices] = True
                contact_regions = (
                    palm_region,
                    np.isin(dominant_joint, finger_joint_groups[0]),
                    np.isin(dominant_joint, finger_joint_groups[4]),
                )
                contact_names = ("palm", "index", "thumb")
                for name, region in zip(contact_names, contact_regions, strict=True):
                    active_region = region & active_height
                    clearance = (
                        float(cup_delta_xy[active_region].min() - cup_radius)
                        if np.any(active_region)
                        else np.inf
                    )
                    if clearance > contact_gap:
                        raise RuntimeError(
                            f"Cup lacks {name} contact at grip frame {frame}: "
                            f"clearance={clearance:.4f} m"
                        )
                cup_direction = cup_positions[frame, :2] - skin[palm_vertex, :2]
                cup_direction /= np.linalg.norm(cup_direction)
                palm_alignment = float(cup_direction @ palm_directions[frame])
                if palm_alignment < minimum_inner_palm_alignment:
                    raise RuntimeError(
                        f"Cup is not on the inner-palm side at grip frame {frame}: "
                        f"alignment={palm_alignment:.4f}"
                    )

        if contact_start_frame < frame < grip_frame:
            clearance = hand_clearance(skin, cup_positions[frame])
            if clearance > 0.001:
                raise RuntimeError(
                    f"Cup moves without hand contact at frame {frame}: "
                    f"clearance={clearance:.4f} m"
                )

        if frame > contact_start_frame:
            cup_relative = cup_positions[frame, :2] - table_center_xy
            overlaps_xy = (
                abs(cup_relative @ axis_x) < half_length + cup_footprint_radius
                and abs(cup_relative @ axis_y) < half_width + cup_footprint_radius
            )
            overlaps_z = (
                cup_positions[frame, 2] - 0.5 * cup_height < table_top_z
                and cup_positions[frame, 2] + 0.5 * cup_height
                > table_top_z - table_thickness
            )
            if overlaps_xy and overlaps_z:
                raise RuntimeError(f"Cup intersects the table at frame {frame}")

    return StaticScene(
        table_center_xy=table_center_xy,
        table_axis_x=axis_x,
        table_axis_y=axis_y,
        table_top_z=table_top_z,
        table_length=table_length,
        table_width=table_width,
        table_thickness=table_thickness,
        cup_center=cup_center,
        cup_positions=cup_positions,
        cup_quaternions=cup_quaternions,
        cup_height=cup_height,
        cup_radius=cup_radius,
        index_contact=index_contact,
        middle_contact=middle_contact,
        contact_frame=contact_frame,
        pickup_frame=contact_start_frame,
        grip_frame=grip_frame,
        cup_is_static=False,
    )


def full_pose(result: np.lib.npyio.NpzFile) -> np.ndarray:
    """Assemble poses in the same 55-joint order used by BodyModel."""
    frames = len(result["trans"])

    def value(name: str, width: int) -> np.ndarray:
        if name not in result.files:
            return np.zeros((frames, width), dtype=np.float64)
        array = np.asarray(result[name], dtype=np.float64)
        if array.shape != (frames, width):
            raise ValueError(
                f"{name} must have shape {(frames, width)}, got {array.shape}"
            )
        return array

    pose = np.concatenate(
        (
            value("root_orient", 3),
            value("pose_body", 63),
            value("pose_jaw", 3),
            value("pose_eye", 6),
            value("pose_hand", 90),
        ),
        axis=1,
    )
    return pose.reshape(frames, 55, 3)


def shaped_template_and_joints(
    result: np.lib.npyio.NpzFile, model: np.lib.npyio.NpzFile
) -> tuple[np.ndarray, np.ndarray]:
    betas = np.asarray(result["betas"], dtype=np.float64).reshape(-1)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    if len(betas) > shapedirs.shape[2]:
        raise ValueError(
            f"Result has {len(betas)} betas, but model only has "
            f"{shapedirs.shape[2]} shape directions"
        )
    vertices = np.asarray(model["v_template"], dtype=np.float64).copy()
    vertices += np.einsum("vcn,n->vc", shapedirs[:, :, : len(betas)], betas)
    regressor = np.asarray(model["J_regressor"], dtype=np.float64)
    joints = regressor @ vertices
    return vertices, joints


def global_joint_motion(
    poses: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return global joint positions and rotations in SMPL-X coordinates."""
    local_rotations = axis_angle_to_matrix(poses)
    frames, joint_count = local_rotations.shape[:2]
    if len(rest_joints) != joint_count or len(parents) != joint_count:
        raise ValueError(
            f"Pose/model joint mismatch: pose={joint_count}, "
            f"joints={len(rest_joints)}, parents={len(parents)}"
        )

    rotations = np.empty_like(local_rotations)
    positions = np.empty((frames, joint_count, 3), dtype=np.float64)
    rotations[:, 0] = local_rotations[:, 0]
    positions[:, 0] = rest_joints[0]

    for joint in range(1, joint_count):
        parent = int(parents[joint])
        if parent < 0 or parent >= joint:
            raise ValueError(f"Invalid parent {parent} for joint {joint}")
        rotations[:, joint] = rotations[:, parent] @ local_rotations[:, joint]
        rest_offset = rest_joints[joint] - rest_joints[parent]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "fij,j->fi", rotations[:, parent], rest_offset
        )

    positions += translations[:, None, :]
    return positions, rotations


def prepare_motion(
    result: np.lib.npyio.NpzFile,
    model_data: np.lib.npyio.NpzFile,
    keep_world_position: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices, rest_joints = shaped_template_and_joints(result, model_data)
    parents = np.asarray(model_data["kintree_table"][0], dtype=np.int64).copy()
    parents[0] = -1
    translations = np.asarray(result["trans"], dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(f"trans must have shape (frames, 3), got {translations.shape}")

    positions, rotations = global_joint_motion(
        full_pose(result), rest_joints, parents, translations
    )

    vertices_mj = vertices @ SMPLX_TO_MUJOCO.T
    rest_joints_mj = rest_joints @ SMPLX_TO_MUJOCO.T
    positions_mj = positions @ SMPLX_TO_MUJOCO.T
    rotations_mj = np.einsum(
        "ab,fjbc,cd->fjad",
        SMPLX_TO_MUJOCO,
        rotations,
        SMPLX_TO_MUJOCO.T,
    )

    if not keep_world_position:
        horizontal_offset = np.array(
            [-positions_mj[0, 0, 0], -positions_mj[0, 0, 1], 0.0]
        )
        positions_mj += horizontal_offset[None, None, :]

    quaternions_mj = matrices_to_quaternions(rotations_mj)
    return vertices_mj, rest_joints_mj, positions_mj, quaternions_mj


def float_values(values: np.ndarray, precision: int = 8) -> str:
    return " ".join(f"{float(value):.{precision}g}" for value in values.reshape(-1))


def int_values(values: np.ndarray) -> str:
    return " ".join(str(int(value)) for value in values.reshape(-1))


def static_scene_xml(scene: StaticScene) -> str:
    yaw = np.arctan2(scene.table_axis_x[1], scene.table_axis_x[0])
    table_quaternion = np.array(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)]
    )
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

    return f'''
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
    '''


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


def effective_frame_rate(result: np.lib.npyio.NpzFile, frame_count: int) -> float:
    stored_fps = float(np.asarray(result["mocap_frame_rate"]).reshape(()))
    if stored_fps <= 0:
        raise ValueError(f"Invalid mocap_frame_rate: {stored_fps}")
    if "mocap_time_length" not in result.files:
        return stored_fps
    duration = float(np.asarray(result["mocap_time_length"]).reshape(()))
    inferred_fps = frame_count / duration if duration > 0 else stored_fps
    return inferred_fps if abs(stored_fps - inferred_fps) > 1.0 else stored_fps


def mocap_ids(model: mujoco.MjModel, joint_count: int) -> np.ndarray:
    ids = np.empty(joint_count, dtype=np.int32)
    for joint in range(joint_count):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"smplx_joint_{joint:02d}"
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body for SMPL-X joint {joint} was not compiled")
        ids[joint] = model.body_mocapid[body_id]
    if np.any(ids < 0):
        raise RuntimeError("One or more SMPL-X joint bodies are not mocap bodies")
    return ids


def cup_mocap_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "massless_cup"
    )
    if body_id < 0 or model.body_mocapid[body_id] < 0:
        raise RuntimeError("The weightless cup was not compiled as a mocap body")
    return int(model.body_mocapid[body_id])


def set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    cup_id: int,
    cup_positions: np.ndarray,
    cup_quaternions: np.ndarray,
    frame: int,
) -> None:
    data.mocap_pos[ids] = positions[frame]
    data.mocap_quat[ids] = quaternions[frame]
    data.mocap_pos[cup_id] = cup_positions[frame]
    data.mocap_quat[cup_id] = cup_quaternions[frame]
    data.time = frame * model.opt.timestep
    mujoco.mj_forward(model, data)


def run_viewer(
    model: mujoco.MjModel,
    positions: np.ndarray,
    quaternions: np.ndarray,
    scene: StaticScene,
    start_frame: int,
    speed: float,
    loop: bool,
) -> None:
    data = mujoco.MjData(model)
    ids = mocap_ids(model, positions.shape[1])
    cup_id = cup_mocap_id(model)
    frame_count = len(positions)
    state = {"frame": start_frame, "paused": False, "loop": loop, "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == 32:  # Space
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):  # R/r
            state["frame"] = start_frame
            state["step"] = 0
        elif keycode == 262:  # Right arrow
            state["paused"] = True
            state["step"] = 1
        elif keycode == 263:  # Left arrow
            state["paused"] = True
            state["step"] = -1
        elif keycode in (76, 108):  # L/l
            state["loop"] = not state["loop"]

    set_frame(
        model,
        data,
        ids,
        positions,
        quaternions,
        cup_id,
        scene.cup_positions,
        scene.cup_quaternions,
        state["frame"],
    )
    frame_duration = model.opt.timestep / speed
    accumulator = 0.0
    previous_time = time.monotonic()

    print("Controls: Space pause/resume | Left/Right step | R restart | L toggle loop")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=True
    ) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.35, 1.15)
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 20.0
        viewer.cam.elevation = -8.0
        displayed_overlay: tuple[int, bool] | None = None

        while viewer.is_running():
            now = time.monotonic()
            elapsed = min(now - previous_time, 0.1)
            previous_time = now

            if state["step"]:
                state["frame"] = int(
                    np.clip(state["frame"] + state["step"], 0, frame_count - 1)
                )
                state["step"] = 0
                accumulator = 0.0
            elif not state["paused"]:
                accumulator += elapsed
                if accumulator >= frame_duration:
                    advance = int(accumulator / frame_duration)
                    accumulator -= advance * frame_duration
                    next_frame = state["frame"] + advance
                    if next_frame >= frame_count:
                        if state["loop"]:
                            loop_length = frame_count - start_frame
                            next_frame = start_frame + (
                                (next_frame - start_frame) % loop_length
                            )
                        else:
                            next_frame = frame_count - 1
                            state["paused"] = True
                    state["frame"] = next_frame

            set_frame(
                model,
                data,
                ids,
                positions,
                quaternions,
                cup_id,
                scene.cup_positions,
                scene.cup_quaternions,
                state["frame"],
            )
            overlay_state = (state["frame"], state["paused"])
            if overlay_state != displayed_overlay:
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Frame (0-based)\nPlayback",
                        f"{state['frame']} / {frame_count - 1}\n"
                        f"{'Paused' if state['paused'] else 'Running'}",
                    )
                )
                displayed_overlay = overlay_state
            viewer.sync()
            time.sleep(0.001)


def main() -> int:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero")
    if args.grip_transition_frames < 1:
        raise ValueError("--grip-transition-frames must be at least one")
    scene_dimensions = (
        args.table_length,
        args.table_width,
        args.table_thickness,
        args.cup_height,
        args.cup_radius,
    )
    if any(value <= 0 for value in scene_dimensions):
        raise ValueError("Table and cup dimensions must all be greater than zero")
    if args.table_thickness >= 0.5 * min(args.table_length, args.table_width):
        raise ValueError("--table-thickness is too large for the tabletop")

    result_path = args.result.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Stage-II result not found: {result_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model not found: {model_path}")

    with np.load(result_path, allow_pickle=True) as result, np.load(
        model_path, allow_pickle=True
    ) as model_data:
        require_keys(result, REQUIRED_RESULT_KEYS, "Stage-II result")
        require_keys(model_data, REQUIRED_MODEL_KEYS, "SMPL-X model")
        positions_source = np.asarray(result["trans"])
        frame_count = len(positions_source)
        if frame_count < 1:
            raise ValueError("Stage-II result contains no frames")
        if not 0 <= args.start_frame < frame_count:
            raise ValueError(
                f"--start-frame must be between 0 and {frame_count - 1}"
            )
        if not 0 <= args.contact_frame < frame_count:
            raise ValueError(
                f"--contact-frame must be between 0 and {frame_count - 1}"
            )
        if not args.contact_frame <= args.pickup_frame < frame_count:
            raise ValueError(
                "--pickup-frame must be at or after --contact-frame and "
                f"no greater than {frame_count - 1}"
            )
        fps = effective_frame_rate(result, frame_count)
        vertices, bind_positions, positions, quaternions = prepare_motion(
            result, model_data, args.keep_world_position
        )
        faces = np.asarray(model_data["f"], dtype=np.int32)
        weights = np.asarray(model_data["weights"], dtype=np.float64)
        scene = build_static_scene(
            result,
            vertices,
            faces,
            bind_positions,
            weights,
            positions,
            quaternions,
            args.table_length,
            args.table_width,
            args.table_thickness,
            args.cup_height,
            args.cup_radius,
            args.contact_frame,
            args.pickup_frame,
            args.grip_transition_frames,
            not args.keep_cup_static,
        )

    print(f"Result: {result_path}")
    print(
        f"Motion: {frame_count} frames at {fps:.3f} fps | "
        f"Skin: {len(vertices)} vertices, {len(faces)} faces, {len(bind_positions)} bones"
    )
    if scene.cup_is_static:
        bottom_clearance_mm = 1000.0 * (
            scene.cup_center[2] - 0.5 * scene.cup_height - scene.table_top_z
        )
        print(
            "Scene: contact frame={} | cup remains stationary on table | "
            "table top z={:.3f} m | cup bottom clearance={:.1f} mm | "
            "index/middle z=({:.3f}, {:.3f}) m".format(
                scene.contact_frame,
                scene.table_top_z,
                bottom_clearance_mm,
                scene.index_contact[2],
                scene.middle_contact[2],
            )
        )
    else:
        bottom_clearance_mm = 1000.0 * (
            scene.cup_center[2] - 0.5 * scene.cup_height - scene.table_top_z
        )
        print(
            "Scene: contact frame={} | pickup after frame={} | palm/index/thumb "
            "contact at frame={} | table top z={:.3f} m | cup bottom "
            "clearance={:.1f} mm | "
            "index/middle z=({:.3f}, {:.3f}) m".format(
                scene.contact_frame,
                scene.pickup_frame,
                scene.grip_frame,
                scene.table_top_z,
                bottom_clearance_mm,
                scene.index_contact[2],
                scene.middle_contact[2],
            )
        )
    print("Compiling MuJoCo skin...")
    model = make_model(vertices, faces, weights, bind_positions, fps, scene)
    print(
        f"Compiled: nskin={model.nskin}, nskinvert={model.nskinvert}, "
        f"nskinbone={model.nskinbone}"
    )

    if args.validate_only:
        data = mujoco.MjData(model)
        ids = mocap_ids(model, positions.shape[1])
        cup_id = cup_mocap_id(model)
        set_frame(
            model,
            data,
            ids,
            positions,
            quaternions,
            cup_id,
            scene.cup_positions,
            scene.cup_quaternions,
            args.start_frame,
        )
        print(f"Validation passed at frame {args.start_frame}; GUI was not opened.")
        return 0

    run_viewer(
        model,
        positions,
        quaternions,
        scene,
        args.start_frame,
        args.speed,
        not args.no_loop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
