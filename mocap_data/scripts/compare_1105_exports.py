#!/usr/bin/env python3
"""Visualize one OptiTrack CSV take from data/1105/csv in MuJoCo.

Only one CSV file is loaded at a time. Bone positions are shown as points and
connected with the standard Motive parent-child skeleton hierarchy. Missing
samples are hidden rather than interpolated so gaps in the export stay visible.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "1105" / "csv"
HIDDEN_POSITION = np.array([0.0, 0.0, -100.0])

CATEGORY_STYLES = {
    "Bone": (0.014, "0.1 0.65 1 1"),
    "Skeleton": (0.009, "0.08 0.48 1 0.9"),
    "Bone Marker": (0.012, "0.15 1 0.35 0.9"),
    "Rigid Body Marker": (0.018, "1 0.45 0.05 1"),
    "Marker": (0.006, "0.75 0.8 0.85 0.38"),
}
CATEGORY_KEYS = {
    49: "Bone",
    50: "Bone Marker",
    51: "Rigid Body",
    52: "Rigid Body Marker",
    53: "Marker",
    54: "Skeleton",
}

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
for side in ("L", "R"):
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        SKELETON_PARENTS[f"{side}{finger}1"] = f"{side}Hand"
        SKELETON_PARENTS[f"{side}{finger}2"] = f"{side}{finger}1"
        SKELETON_PARENTS[f"{side}{finger}3"] = f"{side}{finger}2"


@dataclass
class CsvExport:
    path: Path
    times: np.ndarray
    frame_rate: float
    point_groups: dict[str, tuple[list[str], np.ndarray]]
    rigid_names: list[str]
    rigid_positions: np.ndarray
    rigid_quaternions: np.ndarray
    bone_names: list[str]
    bone_positions: np.ndarray


@dataclass
class Entity:
    name: str
    category: str
    positions: np.ndarray
    quaternions: np.ndarray
    geom_xml: str


def parse_csv_groups(header: list[list[str]]) -> list[dict[str, object]]:
    """Parse consecutive object groups from the seven-row Motive header."""
    types, names, ids, quantities, axes = header[2:7]
    groups: list[dict[str, object]] = []
    column = 2
    while column < len(types):
        key = (types[column].strip(), names[column].strip(), ids[column].strip())
        end = column + 1
        while end < len(types):
            candidate = (
                types[end].strip(),
                names[end].strip(),
                ids[end].strip(),
            )
            if candidate != key:
                break
            end += 1
        groups.append(
            {
                "start": column,
                "end": end,
                "type": key[0],
                "name": key[1],
                "quantities": quantities[column:end],
                "axes": axes[column:end],
            }
        )
        column = end
    return groups


def load_csv_export(path: Path) -> CsvExport:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(7)]
    metadata = dict(zip(header[0][0::2], header[0][1::2]))
    groups = parse_csv_groups(header)
    table = pd.read_csv(path, skiprows=7, header=None, na_values=[""])
    times = table.iloc[:, 1].to_numpy(float)

    points: dict[str, tuple[list[str], np.ndarray]] = {}
    for group_type in ("Bone Marker", "Rigid Body Marker", "Marker"):
        selected = [group for group in groups if group["type"] == group_type]
        names = [str(group["name"]) for group in selected]
        values = [
            table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
            for group in selected
        ]
        if any(value.shape[1] != 3 for value in values):
            raise ValueError(f"{group_type} does not contain XYZ positions")
        positions = (
            np.stack(values, axis=1)
            if values
            else np.empty((len(table), 0, 3), dtype=float)
        )
        points[group_type] = (names, positions)

    bones = [group for group in groups if group["type"] == "Bone"]
    bone_values = [
        table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
        for group in bones
    ]
    if any(value.shape[1] != 7 for value in bone_values):
        raise ValueError("CSV bones must contain quaternion XYZW and position XYZ")
    bone_positions = (
        np.stack([value[:, 4:7] for value in bone_values], axis=1)
        if bone_values
        else np.empty((len(table), 0, 3), dtype=float)
    )

    rigid = [group for group in groups if group["type"] == "Rigid Body"]
    rigid_values = [
        table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
        for group in rigid
    ]
    if any(value.shape[1] != 7 for value in rigid_values):
        raise ValueError(
            "CSV rigid bodies must contain quaternion XYZW and position XYZ"
        )
    if rigid_values:
        rigid_array = np.stack(rigid_values, axis=1)
        rigid_positions = rigid_array[:, :, 4:7]
        rigid_quaternions = rigid_array[:, :, :4][:, :, [3, 0, 1, 2]]
    else:
        rigid_positions = np.empty((len(table), 0, 3), dtype=float)
        rigid_quaternions = np.empty((len(table), 0, 4), dtype=float)

    fallback_rate = 1.0 / float(np.median(np.diff(times)))
    return CsvExport(
        path=path,
        times=times,
        frame_rate=float(metadata.get("Export Frame Rate", fallback_rate)),
        point_groups=points,
        rigid_names=[str(group["name"]) for group in rigid],
        rigid_positions=rigid_positions,
        rigid_quaternions=rigid_quaternions,
        bone_names=[str(group["name"]) for group in bones],
        bone_positions=bone_positions,
    )


def identity_quaternions(frame_count: int, object_count: int) -> np.ndarray:
    result = np.zeros((frame_count, object_count, 4))
    result[:, :, 0] = 1.0
    return result


def z_axis_quaternions(vectors: np.ndarray) -> np.ndarray:
    """Return MuJoCo WXYZ quaternions rotating local +Z onto each vector."""
    result = np.full((len(vectors), 4), np.nan)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (lengths > 1e-9)
    directions = vectors[valid] / lengths[valid, None]
    regular = directions[:, 2] > -1.0 + 1e-8
    quaternions = np.empty((len(directions), 4))
    scale = np.sqrt(2.0 * (1.0 + directions[regular, 2]))
    quaternions[regular, 0] = scale / 2.0
    quaternions[regular, 1] = -directions[regular, 1] / scale
    quaternions[regular, 2] = directions[regular, 0] / scale
    quaternions[regular, 3] = 0.0
    # A 180-degree rotation around X maps +Z to -Z.
    quaternions[~regular] = np.array([0.0, 1.0, 0.0, 0.0])
    result[valid] = quaternions
    return result


def make_skeleton_entities(csv_export: CsvExport) -> list[Entity]:
    """Create fixed-length capsules joining Motive skeleton parent/child bones."""
    entities: list[Entity] = []
    indices = {name: index for index, name in enumerate(csv_export.bone_names)}
    root_by_namespace: dict[str, str] = {}
    for name in csv_export.bone_names:
        namespace, separator, local_name = name.rpartition(":")
        if separator and local_name == namespace:
            root_by_namespace[namespace] = name

    radius, rgba = CATEGORY_STYLES["Skeleton"]
    for child_name, child_index in indices.items():
        namespace, separator, local_name = child_name.rpartition(":")
        if not separator:
            continue
        parent_local = SKELETON_PARENTS.get(local_name)
        if local_name in ("Ab", "LThigh", "RThigh"):
            parent_name = root_by_namespace.get(namespace)
        elif parent_local is not None:
            parent_name = f"{namespace}:{parent_local}"
        else:
            continue
        # A hand-only Motive skeleton can export its wrist as the namespace
        # root without a separate LHand/RHand bone.
        if (
            parent_name not in indices
            and parent_local in ("LHand", "RHand")
            and local_name.endswith("1")
        ):
            parent_name = root_by_namespace.get(namespace)
        if parent_name not in indices:
            continue

        parent_positions = csv_export.bone_positions[:, indices[parent_name]]
        child_positions = csv_export.bone_positions[:, child_index]
        vectors = child_positions - parent_positions
        lengths = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & (lengths > 1e-9)
        if not valid.any():
            continue
        length = float(np.median(lengths[valid]))
        entities.append(
            Entity(
                name=f"{parent_name} -> {child_name}",
                category="Skeleton",
                positions=(parent_positions + child_positions) / 2.0,
                quaternions=z_axis_quaternions(vectors),
                geom_xml=(
                    f'<geom type="capsule" size="{radius:.5g} {length / 2:.7g}" '
                    f'rgba="{rgba}" contype="0" conaffinity="0"/>'
                ),
            )
        )
    return entities


def make_entities(csv_export: CsvExport) -> list[Entity]:
    """Create skeleton links, point tracks, and rigid-body entities."""
    entities: list[Entity] = make_skeleton_entities(csv_export)

    def add_points(
        category: str,
        names: list[str],
        positions: np.ndarray,
    ) -> None:
        radius, rgba = CATEGORY_STYLES[category]
        quaternions = identity_quaternions(len(csv_export.times), len(names))
        for index, name in enumerate(names):
            entities.append(
                Entity(
                    name=name,
                    category=category,
                    positions=positions[:, index],
                    quaternions=quaternions[:, index],
                    geom_xml=(
                        f'<geom type="sphere" size="{radius:.5g}" '
                        f'rgba="{rgba}" contype="0" conaffinity="0"/>'
                    ),
                )
            )

    add_points("Bone", csv_export.bone_names, csv_export.bone_positions)
    for category, (names, positions) in csv_export.point_groups.items():
        add_points(category, names, positions)

    for index, name in enumerate(csv_export.rigid_names):
        entities.append(
            Entity(
                name=name,
                category="Rigid Body",
                positions=csv_export.rigid_positions[:, index],
                quaternions=csv_export.rigid_quaternions[:, index],
                geom_xml=(
                    '<geom type="box" size="0.085 0.055 0.035" '
                    'rgba="1 0.15 0.1 0.85" contype="0" conaffinity="0"/>'
                ),
            )
        )
    return entities


def build_model(entities: list[Entity], timestep: float) -> mujoco.MjModel:
    bodies = []
    for index, entity in enumerate(entities):
        body_name = escape(f"entity_{index:03d}_{entity.category}_{entity.name}")
        bodies.append(
            f'''<body name="{body_name}" mocap="true">
                   {entity.geom_xml}
                </body>'''
        )
    xml = f"""
    <mujoco model="1105_csv_take">
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
        <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def update_scene(
    data: mujoco.MjData,
    entities: list[Entity],
    frame: int,
    visible: dict[str, bool],
) -> None:
    for index, entity in enumerate(entities):
        position = entity.positions[frame]
        quaternion = entity.quaternions[frame]
        if (
            not visible[entity.category]
            or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all()
        ):
            data.mocap_pos[index] = HIDDEN_POSITION
            data.mocap_quat[index] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            data.mocap_pos[index] = position
            data.mocap_quat[index] = quaternion


def print_summary(csv_export: CsvExport) -> None:
    print(f"CSV: {csv_export.path}")
    print(
        f"Frames: {len(csv_export.times)} @ {csv_export.frame_rate:g} Hz; "
        f"duration: {csv_export.times[-1] - csv_export.times[0]:.3f} s"
    )
    skeleton_links = make_skeleton_entities(csv_export)
    print(
        f"Bone: {len(csv_export.bone_names)} points; "
        f"Skeleton: {len(skeleton_links)} parent-child links"
    )
    for category, (names, values) in csv_export.point_groups.items():
        valid = np.isfinite(values).all(axis=2)
        coverage = 100.0 * valid.mean() if valid.size else 0.0
        print(
            f"{category}: {len(names)} tracks; "
            f"visible samples={valid.sum()}/{valid.size} ({coverage:.1f}%)"
        )
    print(f"Rigid Body: {len(csv_export.rigid_names)}")
    print(
        "Colors: Bone/Skeleton=blue, Bone Marker=green, Rigid Body=red, "
        "Rigid Body Marker=orange, Marker=gray"
    )


def available_takes(csv_dir: Path) -> list[Path]:
    return sorted(csv_dir.glob("*.csv"))


def resolve_take(csv_dir: Path, selector: str) -> Path:
    supplied_path = Path(selector).expanduser()
    if supplied_path.is_file():
        return supplied_path.resolve()

    takes = available_takes(csv_dir)
    if not takes:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")
    if selector.isdigit() and 1 <= int(selector) <= len(takes):
        return takes[int(selector) - 1]

    exact = [path for path in takes if path.name == selector or path.stem == selector]
    matches = exact or [
        path for path in takes if selector.lower() in path.stem.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Take selector {selector!r} matches multiple CSV files")
    raise ValueError(f"Cannot resolve take selector {selector!r}; use --list")


def visualize(args: argparse.Namespace) -> None:
    if args.list:
        for index, path in enumerate(available_takes(args.csv_dir), start=1):
            print(f"{index}: {path.name}")
        return

    csv_path = resolve_take(args.csv_dir, args.take)
    csv_export = load_csv_export(csv_path)
    print_summary(csv_export)

    entities = make_entities(csv_export)
    model = build_model(entities, 1.0 / csv_export.frame_rate)
    data = mujoco.MjData(model)
    visible = {category: True for category in CATEGORY_KEYS.values()}
    update_scene(data, entities, 0, visible)
    mujoco.mj_forward(model, data)
    print(f"MuJoCo mocap bodies: {model.nmocap}")
    if args.check_only:
        print("Validation complete; GUI was not opened (--check-only).")
        return

    from mujoco import viewer as mjviewer

    state = {"paused": False, "reset": False, "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):
            state["reset"] = True
        elif keycode in CATEGORY_KEYS:
            category = CATEGORY_KEYS[keycode]
            visible[category] = not visible[category]
            print(f"{category}: {'shown' if visible[category] else 'hidden'}")
        elif keycode == 262:
            state["paused"] = True
            state["step"] = 1
        elif keycode == 263:
            state["paused"] = True
            state["step"] = -1

    frame = 0
    fractional_frames = 0.0
    previous_time = time.perf_counter()
    finite_bones = csv_export.bone_positions[
        np.isfinite(csv_export.bone_positions).all(axis=2)
    ]
    center = (
        np.median(finite_bones, axis=0)
        if len(finite_bones)
        else np.array([0.0, 0.0, 1.0])
    )
    span = (
        float(np.ptp(finite_bones, axis=0).max())
        if len(finite_bones)
        else 2.0
    )

    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(3.5, span * 1.5)
        viewer.cam.azimuth = 115
        viewer.cam.elevation = -18
        print(
            "Controls: Space pause/resume | R restart | Left/Right step | "
            "1 Bone | 2 Bone Marker | 3 Rigid Body | "
            "4 Rigid Body Marker | 5 Marker | 6 Skeleton | Esc close"
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
                frame = int(
                    np.clip(frame + state["step"], 0, len(csv_export.times) - 1)
                )
                state["step"] = 0
            elif not state["paused"]:
                fractional_frames += elapsed * csv_export.frame_rate * args.speed
                advance = int(fractional_frames)
                fractional_frames -= advance
                if advance:
                    frame += advance
                    if frame >= len(csv_export.times):
                        if args.no_loop:
                            frame = len(csv_export.times) - 1
                            state["paused"] = True
                        else:
                            frame %= len(csv_export.times)

            update_scene(data, entities, frame, visible)
            data.time = csv_export.times[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.001)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "take",
        nargs="?",
        default="1",
        help=(
            "1-based CSV number, exact name, unique substring, or CSV path "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="directory containing the available CSV takes",
    )
    parser.add_argument("--list", action="store_true", help="list available CSV takes")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument("--no-loop", action="store_true", help="stop on the final frame")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="parse and validate without opening a GUI",
    )
    return parser


if __name__ == "__main__":
    visualize(make_parser().parse_args())
