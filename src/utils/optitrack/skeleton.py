"""Reusable optitrack skeleton helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils.optitrack.c3d import (
    C3DMotion,
)

# Parent names used by Motive's conventional full-body skeleton. Each CSV bone
# carries a namespace (for example ``Skeleton 003:``); connections are made
# only between bones in the same namespace.
SKELETON_PARENTS = {
    "Ab": None,
    "Chest": "Ab",
    "Neck": "Chest",
    "Head": "Neck",
    "LShoulder": "Chest",
    "LUArm": "LShoulder",
    "LFArm": "LUArm",
    "LHand": "LFArm",
    "RShoulder": "Chest",
    "RUArm": "RShoulder",
    "RFArm": "RUArm",
    "RHand": "RFArm",
    "LThigh": None,
    "LShin": "LThigh",
    "LFoot": "LShin",
    "LToe": "LFoot",
    "RThigh": None,
    "RShin": "RThigh",
    "RFoot": "RShin",
    "RToe": "RFoot",
}


for _side in ("L", "R"):
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        SKELETON_PARENTS[f"{_side}{_finger}1"] = f"{_side}Hand"
        SKELETON_PARENTS[f"{_side}{_finger}2"] = f"{_side}{_finger}1"
        SKELETON_PARENTS[f"{_side}{_finger}3"] = f"{_side}{_finger}2"


# Virtual upper-body joints derived only from the named surface markers in the
# 25-marker layout. These are for visualization and are not anatomical joint
# centers or inputs to MoSh++ optimization.
SKELETON_NODE_MARKERS = {
    "pelvis": ("WaistLFront", "WaistLBack", "WaistRFront", "WaistRBack"),
    "spine": ("BackLeft", "BackRight", "Chest"),
    "neck": ("BackTop",),
    "head": ("HeadTop", "HeadFront", "HeadSide"),
    "left_shoulder": ("LShoulderBack", "LShoulderTop"),
    "left_upper_arm": ("LUArmHigh",),
    "left_elbow": ("LElbowOut",),
    "left_wrist": ("LWristOut", "LWristIn"),
    "left_hand": ("LHandOut",),
    "right_shoulder": ("RShoulderBack", "RShoulderTop"),
    "right_upper_arm": ("RUArmHigh",),
    "right_elbow": ("RElbowOut",),
    "right_wrist": ("RWristOut", "RWristIn"),
    "right_hand": ("RHandOut",),
}


SKELETON_EDGES = (
    ("pelvis", "spine"),
    ("spine", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_upper_arm"),
    ("left_upper_arm", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_upper_arm"),
    ("right_upper_arm", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)


@dataclass(frozen=True)
class SkeletonMotion:
    node_names: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    edges: tuple[tuple[int, int], ...]


def display_labels(labels: Sequence[str], short_labels: bool) -> tuple[str, ...]:
    if not short_labels:
        return tuple(labels)
    return tuple(label.split(":", 1)[-1] for label in labels)


def build_skeleton_motion(motion: C3DMotion) -> SkeletonMotion | None:
    local_indices = {
        label.split(":", 1)[-1]: index for index, label in enumerate(motion.labels)
    }
    required = {
        marker for markers in SKELETON_NODE_MARKERS.values() for marker in markers
    }
    missing = sorted(required.difference(local_indices))
    if missing:
        print(
            "Skeleton disabled because required marker labels are missing: "
            + ", ".join(missing)
        )
        return None

    node_names = tuple(SKELETON_NODE_MARKERS)
    positions = np.empty((len(motion.positions), len(node_names), 3), dtype=float)
    valid = np.empty((len(motion.positions), len(node_names)), dtype=bool)
    for node_index, node_name in enumerate(node_names):
        marker_indices = [
            local_indices[marker] for marker in SKELETON_NODE_MARKERS[node_name]
        ]
        node_valid = motion.valid[:, marker_indices].all(axis=1)
        node_positions = np.nanmean(motion.positions[:, marker_indices], axis=1)
        node_positions[~node_valid] = np.nan
        positions[:, node_index] = node_positions
        valid[:, node_index] = node_valid
    node_indices = {name: index for index, name in enumerate(node_names)}
    edges = tuple(
        (node_indices[parent], node_indices[child]) for parent, child in SKELETON_EDGES
    )
    return SkeletonMotion(node_names, positions, valid, edges)
