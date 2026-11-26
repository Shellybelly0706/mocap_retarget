"""Reusable smplx scene helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.smplx.kinematics import (
    skinned_vertices_at_frame,
)
from utils.smplx.models import (
    FINGER_LABEL_FALLBACKS,
    StaticScene,
)


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
        [
            cosine * outward[0] - sine * outward[1],
            sine * outward[0] + cosine * outward[1],
        ]
    )
    axis_y = np.array([-axis_x[1], axis_x[0]])

    # Marker vertices are not necessarily the outermost fingertip vertices.
    # This marker-to-corner offset puts the actual skin surface at the corner.
    corner_clearance = 0.012
    vertical_clearance = 0.001
    near_corner = contact_midpoint[:2] + corner_clearance * outward
    table_center_xy = near_corner + half_length * axis_x + half_width * axis_y
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
        in_body = (radial_distance < cup_radius) & (
            np.abs(delta[:, 2]) < cup_half_height
        )
        # The visible rim is a ring of capsules, not a wider cylinder along the
        # full cup height. Treating it separately lets the hand touch the body.
        rim_distance = np.sqrt(
            (radial_distance - cup_radius) ** 2 + (delta[:, 2] - rim_center_height) ** 2
        )
        in_rim = rim_distance < rim_radius
        in_opening = (radial_distance < 0.82 * cup_radius) & (
            np.abs(delta[:, 2] - (cup_half_height + 0.0025)) < 0.0015
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
        active = hand_contact_region & (np.abs(delta[:, 2]) < cup_half_height)
        return float(np.linalg.norm(delta[active, :2], axis=1).min() - cup_radius)

    def table_pose_toward_cup(
        cup_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search a corner-anchored pose containing the cup without body overlap."""
        slab_points: list[np.ndarray] = []
        for skin in skin_frames:
            in_slab_height = (skin[:, 2] < table_top_z) & (
                skin[:, 2] > table_top_z - table_thickness
            )
            shallow_hand_contact = hand_contact_region & (
                skin[:, 2] >= table_top_z - 0.008
            )
            slab_points.append(skin[in_slab_height & ~shallow_hand_contact, :2])
        forbidden_points = np.concatenate(slab_points, axis=0)

        poses: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        original_axis_x = axis_x.copy()
        for angle_degrees in range(0, 360, 2):
            angle = np.deg2rad(angle_degrees)
            candidate_axis_x = np.array([np.cos(angle), np.sin(angle)])
            candidate_axis_y = np.array([-candidate_axis_x[1], candidate_axis_x[0]])
            for corner_x in (-1.0, 1.0):
                for corner_y in (-1.0, 1.0):
                    candidate_center = (
                        near_corner
                        - corner_x * half_length * candidate_axis_x
                        - corner_y * half_width * candidate_axis_y
                    )
                    cup_relative = cup_xy - candidate_center
                    if (
                        abs(cup_relative @ candidate_axis_x) + cup_footprint_radius
                        > half_length
                        or abs(cup_relative @ candidate_axis_y) + cup_footprint_radius
                        > half_width
                    ):
                        continue
                    relative = forbidden_points - candidate_center
                    inside = (np.abs(relative @ candidate_axis_x) < half_length) & (
                        np.abs(relative @ candidate_axis_y) < half_width
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
                candidate[:2] += offset * np.array([np.cos(angle), np.sin(angle)])
                if cup_intersects_skin(swept_hand, candidate):
                    continue
                delta = swept_hand - candidate
                active = np.abs(delta[:, 2]) < cup_half_height
                clearance = float(
                    np.linalg.norm(delta[active, :2], axis=1).min() - cup_radius
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
        cup_positions = np.repeat(cup_center[None, :], len(joint_positions), axis=0)
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
                raise RuntimeError(f"Table intersects the SMPL-X skin at frame {frame}")
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
        (40, 41, 42),  # index
        (43, 44, 45),  # middle
        (46, 47, 48),  # pinky
        (49, 50, 51),  # ring
        (52, 53, 54),  # thumb
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
    marker_distances = np.linalg.norm(vertices - vertices[palm_marker_vertex], axis=1)
    normal_alignment = bind_normals @ bind_normals[palm_marker_vertex]
    opposite_candidates = np.flatnonzero(
        (dominant_joint == 21) & (marker_distances < 0.06) & (normal_alignment < -0.8)
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
            raise RuntimeError(
                "Opposite-palm surface normal has no horizontal component"
            )
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
    vertical_grip_offset = tabletop_cup_z - skin_frames[grip_frame][palm_vertex, 2]
    for frame in range(pickup_frame + 1, len(joint_positions)):
        skin = skin_frames[frame]
        palm = skin[palm_vertex]
        candidate_z = palm[2] + vertical_grip_offset
        nearby = (np.abs(skin[:, 2] - candidate_z) < cup_half_height) & (
            np.linalg.norm(skin[:, :2] - palm[:2], axis=1) < 0.25
        )
        nearby_points = skin[nearby, :2]
        nearby_joints = dominant_joint[nearby]
        palm_patch_points = skin[palm_patch_vertices]
        palm_patch_points = palm_patch_points[
            np.abs(palm_patch_points[:, 2] - candidate_z) < cup_half_height,
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
                    float(distances[group_mask].min()) if np.any(group_mask) else np.inf
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
            other_finger_contacts = sum(gap <= contact_gap for gap in group_gaps[1:4])
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
        transition_cost = (
            np.linalg.norm(
                current_offsets[:, None, :] - previous_offsets[None, :, :],
                axis=2,
            )
            ** 2
        )
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
            in_tabletop & hand_contact_region & (skin[:, 2] >= table_top_z - 0.008)
        )
        invalid_table_collision = (in_tabletop | in_leg) & ~shallow_hand_contact
        if np.any(invalid_table_collision):
            collision_type = (
                "top" if np.any(in_tabletop & ~shallow_hand_contact) else "leg"
            )
            collision_ids = np.flatnonzero(invalid_table_collision)
            maximum_depth = float(np.max(table_top_z - skin[collision_ids, 2]))
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
                np.abs(skin[:, 2] - cup_positions[frame, 2]) < cup_half_height
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
