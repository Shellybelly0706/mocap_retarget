#!/usr/bin/env python3
"""Play a labeled C3D marker capture in MuJoCo.

Every valid C3D point is rendered as a sphere with its point label. Missing
samples stay hidden; trajectories are never interpolated.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_C3D_READER = REPOSITORY_ROOT / "src/moshpp/src/moshpp/tools/c3d.py"
UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
}
HIDDEN_POSITION = np.array([0.0, 0.0, -100.0])
MARKER_RGBA = np.array([0.1, 0.9, 0.3, 1.0], dtype=np.float32)
JOINT_RGBA = np.array([1.0, 0.72, 0.08, 1.0], dtype=np.float32)
BONE_RGBA = np.array([0.08, 0.48, 1.0, 0.95], dtype=np.float32)

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
class C3DMotion:
    path: Path
    labels: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    frame_numbers: np.ndarray
    frame_rate: float
    source_units: str


@dataclass(frozen=True)
class SkeletonMotion:
    node_names: tuple[str, ...]
    positions: np.ndarray
    valid: np.ndarray
    edges: tuple[tuple[int, int], ...]


def load_c3d_module(reader_path: Path):
    if not reader_path.is_file():
        raise FileNotFoundError(f"Bundled C3D reader does not exist: {reader_path}")
    spec = importlib.util.spec_from_file_location("_local_moshpp_c3d", reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load C3D reader: {reader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_label(value: object, index: int) -> str:
    if isinstance(value, bytes):
        label = value.decode("utf-8", errors="replace")
    else:
        label = str(value)
    return label.rstrip("\x00 ").strip() or f"Point_{index + 1}"


def load_c3d(path: Path, reader_path: Path) -> C3DMotion:
    if not path.is_file():
        raise FileNotFoundError(path)
    c3d = load_c3d_module(reader_path)
    with path.open("rb") as stream, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="missing parameter POINT:DESCRIPTIONS")
        warnings.filterwarnings("ignore", message="No analog data found in file.*")
        reader = c3d.Reader(stream)
        labels = tuple(
            clean_label(value, index)
            for index, value in enumerate(reader.point_labels)
        )
        units_parameter = reader.get("POINT:UNITS")
        source_units = (
            units_parameter.string_value.rstrip("\x00 ").strip()
            if units_parameter is not None
            else ""
        )
        scale = UNIT_TO_METERS.get(source_units.lower())
        if scale is None:
            raise ValueError(
                f"Unsupported or missing C3D POINT:UNITS {source_units!r}; "
                f"supported units: {sorted(UNIT_TO_METERS)}"
            )
        frame_rate = float(reader.point_rate)
        frames = list(reader.read_frames(copy=True))

    if not frames:
        raise ValueError(f"C3D has no point frames: {path}")
    positions = np.stack([points[:, :3] for _, points, _ in frames]).astype(float)
    residuals = np.stack([points[:, 3] for _, points, _ in frames])
    valid = np.isfinite(positions).all(axis=2) & (residuals >= 0.0)
    positions *= scale
    positions[~valid] = np.nan
    if positions.shape[1] != len(labels):
        raise ValueError(
            f"C3D has {positions.shape[1]} point tracks but {len(labels)} labels"
        )
    if frame_rate <= 0 or not np.isfinite(frame_rate):
        raise ValueError(f"Invalid C3D point rate: {frame_rate}")
    return C3DMotion(
        path=path,
        labels=labels,
        positions=positions,
        valid=valid,
        frame_numbers=np.asarray([number for number, _, _ in frames], dtype=int),
        frame_rate=frame_rate,
        source_units=source_units,
    )


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


def display_labels(labels: Sequence[str], short_labels: bool) -> tuple[str, ...]:
    if not short_labels:
        return tuple(labels)
    return tuple(label.split(":", 1)[-1] for label in labels)


def build_skeleton_motion(motion: C3DMotion) -> SkeletonMotion | None:
    local_indices = {
        label.split(":", 1)[-1]: index for index, label in enumerate(motion.labels)
    }
    required = {
        marker
        for markers in SKELETON_NODE_MARKERS.values()
        for marker in markers
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
        (node_indices[parent], node_indices[child])
        for parent, child in SKELETON_EDGES
    )
    return SkeletonMotion(node_names, positions, valid, edges)


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


def print_summary(motion: C3DMotion) -> None:
    frame_count, marker_count = motion.positions.shape[:2]
    print(f"C3D: {motion.path}")
    print(
        f"Frames: {frame_count} @ {motion.frame_rate:g} Hz; "
        f"duration: {(frame_count - 1) / motion.frame_rate:.3f} s"
    )
    print(f"Markers: {marker_count}; source units: {motion.source_units}; viewer units: m")
    print(
        f"Valid samples: {int(motion.valid.sum())}/{motion.valid.size} "
        f"({100.0 * motion.valid.mean():.1f}%)"
    )
    print("Marker labels:")
    for index, label in enumerate(motion.labels, start=1):
        print(f"  {index:02d}: {label}")


def visualize(args: argparse.Namespace) -> None:
    motion = load_c3d(args.c3d.expanduser().resolve(), args.c3d_reader.resolve())
    labels = display_labels(motion.labels, args.short_labels)
    skeleton = None if args.no_skeleton else build_skeleton_motion(motion)
    print_summary(motion)
    if skeleton is not None:
        print(
            f"Skeleton: {len(skeleton.node_names)} virtual joints, "
            f"{len(skeleton.edges)} connected bone segments"
        )
    if args.check_only:
        print("Validation complete; GUI was not opened (--check-only).")
        return

    finite_positions = motion.positions[motion.valid]
    center = np.median(finite_positions, axis=0)
    span = float(np.ptp(finite_positions, axis=0).max())
    floor_height = float(np.nanmin(motion.positions[:, :, 2]) - 0.05)
    model = build_model(1.0 / motion.frame_rate, floor_height)
    data = mujoco.MjData(model)

    from mujoco import viewer as mjviewer

    state = {
        "paused": False,
        "reset": False,
        "step": 0,
        "labels": True,
        "markers": True,
        "skeleton": skeleton is not None,
    }

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):
            state["reset"] = True
        elif keycode in (76, 108):
            state["labels"] = not state["labels"]
            print(f"Labels: {'shown' if state['labels'] else 'hidden'}")
        elif keycode in (77, 109):
            state["markers"] = not state["markers"]
            print(f"Markers: {'shown' if state['markers'] else 'hidden'}")
        elif keycode in (66, 98) and skeleton is not None:
            state["skeleton"] = not state["skeleton"]
            print(f"Skeleton: {'shown' if state['skeleton'] else 'hidden'}")
        elif keycode == 262:
            state["paused"] = True
            state["step"] = 1
        elif keycode == 263:
            state["paused"] = True
            state["step"] = -1

    frame = 0
    fractional_frames = 0.0
    previous_time = time.perf_counter()
    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(2.0, span * 2.0)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        initialize_marker_geoms(viewer.user_scn, labels, args.marker_size)
        if skeleton is not None:
            initialize_skeleton_geoms(
                viewer.user_scn,
                len(labels),
                skeleton,
                args.joint_size,
                args.bone_width,
            )
        print(
            "Controls: Space pause/resume | R restart | Left/Right step | "
            "L labels | M markers | B skeleton | Esc close"
        )
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = min(now - previous_time, 0.1)
            previous_time = now
            if state["reset"]:
                frame = 0
                fractional_frames = 0.0
                state["reset"] = False
            if state["step"]:
                frame = int(np.clip(frame + state["step"], 0, len(motion.positions) - 1))
                state["step"] = 0
            elif not state["paused"]:
                fractional_frames += elapsed * motion.frame_rate * args.speed
                advance = int(fractional_frames)
                fractional_frames -= advance
                if advance:
                    frame += advance
                    if frame >= len(motion.positions):
                        if args.no_loop:
                            frame = len(motion.positions) - 1
                            state["paused"] = True
                        else:
                            frame %= len(motion.positions)
            update_marker_geoms(
                viewer.user_scn,
                motion,
                labels,
                frame,
                state["labels"],
                state["markers"],
            )
            if skeleton is not None:
                update_skeleton_geoms(
                    viewer.user_scn,
                    len(labels),
                    skeleton,
                    frame,
                    state["skeleton"],
                    args.bone_width,
                )
            data.time = frame / motion.frame_rate
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.001)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c3d", type=Path, help="C3D capture to visualize")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.012,
        help="marker sphere radius in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--joint-size",
        type=float,
        default=0.016,
        help="virtual skeleton joint radius in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--bone-width",
        type=float,
        default=0.008,
        help="skeleton bone radius in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="disable the marker-derived upper-body skeleton",
    )
    parser.add_argument(
        "--short-labels",
        action="store_true",
        help="hide the subject namespace before ':' in displayed labels",
    )
    parser.add_argument("--no-loop", action="store_true", help="stop on the final frame")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="parse and validate without opening a GUI",
    )
    parser.add_argument(
        "--c3d-reader",
        type=Path,
        default=DEFAULT_C3D_READER,
        help="path to MoSh++'s bundled c3d.py reader",
    )
    return parser


if __name__ == "__main__":
    visualize(make_parser().parse_args())
