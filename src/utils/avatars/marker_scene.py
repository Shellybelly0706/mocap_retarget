"""Reusable avatars marker scene helpers."""

from __future__ import annotations

from xml.sax.saxutils import escape

import mujoco
import numpy as np

from utils.avatars.marker_motion import (
    marker_color,
)


def build_model(
    names: list[str],
    initial_positions: np.ndarray,
    reconstructed: set[str],
    marker_size: float,
) -> mujoco.MjModel:
    bodies = []
    for index, (name, position) in enumerate(
        zip(names, initial_positions, strict=True)
    ):
        body_name = escape(f"marker_{index:02d}_{name}")
        pos = " ".join(f"{value:.9g}" for value in position)
        rgba = marker_color(name, reconstructed)
        bodies.append(
            f"""<body name="{body_name}" mocap="true" pos="{pos}">
                   <geom type="sphere" size="{marker_size:.9g}" rgba="{rgba}"
                         contype="0" conaffinity="0"/>
                </body>"""
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
