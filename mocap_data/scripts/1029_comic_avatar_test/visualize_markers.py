#!/usr/bin/env python3
"""Visualize the 25 OptiTrack bone-marker trajectories in MuJoCo.

The CSV contains solved ``Bone Marker`` positions, not raw camera images.  The
HeadTop trajectory is absent from every exported frame, so by default it is
reconstructed from the solved Head pose and the HeadTop constraint recorded in
Nodes.dat.  The reconstructed marker is displayed in magenta.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    PROJECT_ROOT
    / "data"
    / "1029"
    / "Chi 2026-10-29 02.51.19 PM.csv"
)

# HeadTop constraint in the fitted ``test`` skeleton from Nodes.dat, converted
# into the local coordinate convention used by the exported Head quaternion.
# Nodes.dat offset: (1.19209e-7, 0.226815, -0.0163786)
# Exported local basis: (-X, Z, Y)
HEADTOP_LOCAL = np.array([-1.19209e-7, -0.0163786, 0.226815])


def parse_groups(header_rows: list[list[str]]) -> list[dict[str, object]]:
    """Return consecutive OptiTrack column groups from the seven-row header."""
    types, names, ids, quantities, axes = header_rows[2:7]
    groups: list[dict[str, object]] = []
    column = 2  # Frame and Time occupy columns 0 and 1.
    while column < len(types):
        key = (types[column], names[column], ids[column])
        end = column + 1
        while end < len(types) and (types[end], names[end], ids[end]) == key:
            end += 1
        groups.append(
            {
                "start": column,
                "end": end,
                "type": types[column],
                "name": names[column],
                "id": ids[column],
                "quantities": quantities[column:end],
                "axes": axes[column:end],
            }
        )
        column = end
    return groups


def interpolate_marker_gaps(positions: np.ndarray) -> tuple[np.ndarray, int]:
    """Linearly fill finite marker gaps, including nearest endpoint fill."""
    result = positions.copy()
    missing_before = int(np.isnan(result).sum())
    for marker in range(result.shape[1]):
        for axis in range(3):
            values = pd.Series(result[:, marker, axis])
            if values.notna().any():
                result[:, marker, axis] = values.interpolate(
                    method="linear", limit_direction="both"
                ).to_numpy()
    return result, missing_before


def load_marker_motion(csv_path: Path) -> dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [next(reader) for _ in range(7)]

    metadata = dict(zip(header[0][0::2], header[0][1::2]))
    groups = parse_groups(header)
    table = pd.read_csv(csv_path, skiprows=7, header=None, na_values=[""])

    marker_groups = [group for group in groups if group["type"] == "Bone Marker"]
    if len(marker_groups) != 25:
        raise ValueError(
            f"Expected 25 Bone Marker groups, found {len(marker_groups)} in {csv_path}"
        )

    names = [str(group["name"]).split(":", 1)[-1] for group in marker_groups]
    positions = np.stack(
        [
            table.iloc[:, int(group["start"]) : int(group["end"])].to_numpy(float)
            for group in marker_groups
        ],
        axis=1,
    )

    reconstructed: list[str] = []
    headtop_index = names.index("HeadTop")
    if np.isnan(positions[:, headtop_index]).all():
        head_group = next(
            group
            for group in groups
            if group["type"] == "Bone" and str(group["name"]).endswith(":Head")
        )
        head = table.iloc[
            :, int(head_group["start"]) : int(head_group["end"])
        ].to_numpy(float)
        head_rotation = Rotation.from_quat(head[:, :4])  # CSV order is XYZW.
        positions[:, headtop_index] = (
            head_rotation.apply(np.broadcast_to(HEADTOP_LOCAL, (len(table), 3)))
            + head[:, 4:7]
        )
        reconstructed.append("HeadTop")

    positions, missing_values = interpolate_marker_gaps(positions)
    if not np.isfinite(positions).all():
        bad = [names[i] for i in range(len(names)) if not np.isfinite(positions[:, i]).all()]
        raise ValueError(f"Markers remain entirely unavailable after reconstruction: {bad}")

    times = table.iloc[:, 1].to_numpy(float)
    frame_rate = float(metadata.get("Export Frame Rate", 1.0 / np.diff(times).mean()))
    return {
        "names": names,
        "positions": positions,
        "times": times,
        "frame_rate": frame_rate,
        "metadata": metadata,
        "reconstructed": reconstructed,
        "missing_values_filled": missing_values,
    }


def marker_color(name: str, reconstructed: set[str]) -> str:
    if name in reconstructed:
        return "1 0 1 1"  # Magenta: inferred, not recorded.
    if name.startswith("L"):
        return "0.15 0.45 1 1"
    if name.startswith("R"):
        return "1 0.25 0.2 1"
    if name.startswith("Head"):
        return "1 0.8 0.1 1"
    return "0.2 0.9 0.35 1"


def build_model(
    names: list[str],
    initial_positions: np.ndarray,
    reconstructed: set[str],
    marker_size: float,
) -> mujoco.MjModel:
    bodies = []
    for index, (name, position) in enumerate(zip(names, initial_positions, strict=True)):
        body_name = escape(f"marker_{index:02d}_{name}")
        pos = " ".join(f"{value:.9g}" for value in position)
        rgba = marker_color(name, reconstructed)
        bodies.append(
            f'''<body name="{body_name}" mocap="true" pos="{pos}">
                   <geom type="sphere" size="{marker_size:.9g}" rgba="{rgba}"
                         contype="0" conaffinity="0"/>
                </body>'''
        )

    xml = f"""
    <mujoco model="optitrack_marker_cloud">
      <option timestep="0.008333333333" gravity="0 0 -9.81"/>
      <visual>
        <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/>
        <rgba haze="0.15 0.2 0.25 1"/>
      </visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.18 0.18 0.18" rgb2="0.28 0.28 0.28"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
      </asset>
      <worldbody>
        <light pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
        {''.join(bodies)}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def print_summary(motion: dict[str, object], model: mujoco.MjModel) -> None:
    positions = np.asarray(motion["positions"])
    names = list(motion["names"])
    reconstructed = list(motion["reconstructed"])
    print(f"Frames: {len(positions)}")
    print(f"Frame rate: {motion['frame_rate']:.6g} Hz")
    print(f"Duration: {(len(positions) - 1) / float(motion['frame_rate']):.3f} s")
    print(f"Markers: {len(names)} ({', '.join(names)})")
    print(f"MuJoCo mocap bodies: {model.nmocap}")
    print(f"CSV missing coordinate values filled: {motion['missing_values_filled']}")
    print(
        "Reconstructed rather than measured: "
        + (", ".join(reconstructed) if reconstructed else "none")
    )


def visualize(args: argparse.Namespace) -> None:
    motion = load_marker_motion(args.csv)
    names = list(motion["names"])
    positions = np.asarray(motion["positions"])
    reconstructed = set(motion["reconstructed"])
    frame_rate = float(motion["frame_rate"])

    model = build_model(names, positions[0], reconstructed, args.marker_size)
    data = mujoco.MjData(model)
    data.mocap_pos[:] = positions[0]
    data.mocap_quat[:, 0] = 1.0
    mujoco.mj_forward(model, data)
    print_summary(motion, model)

    if args.check_only:
        return

    from mujoco import viewer as mjviewer

    state = {"paused": False, "reset": False}

    def key_callback(keycode: int) -> None:
        if keycode == 32:  # Space
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):  # R/r
            state["reset"] = True

    frame = 0
    fractional_frames = 0.0
    previous_time = time.perf_counter()
    center = positions.mean(axis=(0, 1))

    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(3.0, np.ptp(positions.reshape(-1, 3), axis=0).max() * 1.6)
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
                    if frame >= len(positions):
                        if args.no_loop:
                            frame = len(positions) - 1
                            state["paused"] = True
                        else:
                            frame %= len(positions)

            data.mocap_pos[:] = positions[frame]
            data.time = frame / frame_rate
            mujoco.mj_forward(model, data)
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
    parser.add_argument(
        "--marker-size", type=float, default=0.018, help="sphere radius in meters"
    )
    parser.add_argument("--no-loop", action="store_true", help="stop on the last frame")
    parser.add_argument(
        "--check-only", action="store_true", help="validate data/model without opening a GUI"
    )
    return parser


if __name__ == "__main__":
    visualize(make_parser().parse_args())
