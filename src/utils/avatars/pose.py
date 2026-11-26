"""Reusable avatars pose helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from utils.avatars.marker_motion import parse_groups
from utils.avatars.materials import (
    SHIRT_MATERIAL,
    SKIN_MATERIAL,
    TROUSERS_MATERIAL,
)
from utils.avatars.models import (
    AnatomicalPart,
    AvatarAsset,
)

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


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
        values = table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
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
    source_position = {name: source[index] for name, index in joint_index.items()}
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
        target_hand_frame = limb_frames(target[hand] - target[lower_arm], target_up)
        deformations[:, joint_index[hand]] = target_hand_frame @ source_hand_frame.T

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

    source_height = float(asset.vertices[:, 2].max() - source_position["hips"][2])
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
