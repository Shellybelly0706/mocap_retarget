"""Generate a reusable Neon MJCF model and a small inspection scene."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .camera import load_scene_camera_calibration
from .imu import camera_to_imu, imu_pose_in_cad


def values(array) -> str:
    return " ".join(f"{number:.12g}" for number in np.asarray(array).reshape(-1))


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_neon_model(asset_dir: Path, calibration_path: Path) -> Path:
    """Write neon.xml and scene.xml using shipped meshes and real intrinsics."""
    asset_dir = Path(asset_dir)
    calibration = load_scene_camera_calibration(calibration_path)
    geometry = json.loads((asset_dir / "geometry.json").read_text())
    reference = geometry["camera_reference"]
    rotation = np.asarray(reference["rotation_cad_from_opencv"])
    if not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(
        np.linalg.det(rotation), 1
    ):
        raise ValueError(
            "Camera reference rotation must be a right-handed orthonormal matrix"
        )

    root = ET.Element("mujoco", model="pupil_labs_neon")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(root, "option", gravity="0 0 0")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="neon_module_serial", data=calibration.serial)
    ET.SubElement(
        custom,
        "text",
        name="neon_camera_pose_source",
        data="CAD lens-opening centre; not a calibrated optical-centre transform",
    )
    ET.SubElement(
        custom,
        "text",
        name="neon_camera_projection",
        data="Factory pinhole intrinsics; lens distortion stored but not rendered",
    )
    ET.SubElement(
        custom, "numeric", name="neon_camera_matrix", data=values(calibration.matrix)
    )
    ET.SubElement(
        custom,
        "numeric",
        name="neon_distortion_opencv",
        data=values(calibration.distortion),
    )
    rotation_i_c, translation_i_c = camera_to_imu()
    ET.SubElement(
        custom,
        "numeric",
        name="neon_rotation_imu_from_camera",
        data=values(rotation_i_c),
    )
    ET.SubElement(
        custom,
        "numeric",
        name="neon_translation_imu_from_camera_m",
        data=values(translation_i_c),
    )
    ET.SubElement(
        custom,
        "text",
        name="neon_imu_reference_quality",
        data="Manufacturer nominal transform; CAD origin inherits approximate lens-opening reference",
    )
    asset = ET.SubElement(root, "asset")
    for name in geometry["components"]:
        ET.SubElement(
            asset,
            "mesh",
            name=f"neon_{name}",
            file=f"meshes/{name}.stl",
            maxhullvert="64",
        )
    world = ET.SubElement(root, "worldbody")
    # CAD Y-up -> scene Z-up. Keep the CAD origin and all component placements.
    body = ET.SubElement(
        world,
        "body",
        name="neon_glasses",
        quat="0.707106781186548 0.707106781186548 0 0",
    )
    colours = {
        "frame": "0.09 0.11 0.14 1",
        "module": "0.17 0.19 0.22 1",
        "hardware": "0.42 0.46 0.5 1",
    }
    for name, component in geometry["components"].items():
        ET.SubElement(
            body,
            "geom",
            name=f"neon_{name}_visual",
            type="mesh",
            mesh=f"neon_{name}",
            rgba=colours[component["group"]],
            contype="0",
            conaffinity="0",
            density="0",
            group="1",
        )
    imu_position, imu_rotation = imu_pose_in_cad(reference["position_cad_m"], rotation)
    imu = ET.SubElement(
        body,
        "body",
        name="neon_imu",
        pos=values(imu_position),
        xyaxes=values(np.concatenate([imu_rotation[:, 0], imu_rotation[:, 1]])),
    )
    ET.SubElement(
        imu,
        "site",
        name="neon_imu_origin",
        size="0.0015",
        rgba="1 0.1 0.8 1",
        group="4",
    )
    for axis, end, colour in [
        ("x", [0.018, 0, 0], "1 0.1 0.1 1"),
        ("y", [0, 0.018, 0], "0.1 0.9 0.1 1"),
        ("z", [0, 0, 0.022], "0.1 0.3 1 1"),
    ]:
        ET.SubElement(
            imu,
            "site",
            name=f"neon_imu_{axis}",
            type="capsule",
            fromto="0 0 0 " + values(end),
            size="0.0005",
            rgba=colour,
            group="4",
        )
    optical = ET.SubElement(
        body,
        "body",
        name="neon_scene_camera_optical",
        pos=values(reference["position_cad_m"]),
        xyaxes=values(np.concatenate([rotation[:, 0], rotation[:, 1]])),
    )
    optical.append(ET.Comment("OpenCV optical frame: +X right, +Y down, +Z forward."))
    # Rotate OpenCV to OpenGL by 180 degrees about X, preserving the origin.
    ET.SubElement(
        optical,
        "camera",
        name="neon_scene_camera",
        quat="0 1 0 0",
        **calibration.mjcf_attributes(),
    )
    ET.SubElement(
        optical,
        "site",
        name="neon_scene_camera_origin",
        size="0.001",
        rgba="1 0.8 0.1 1",
        group="5",
    )
    for axis, end, colour in [
        ("x", [0.025, 0, 0], "1 0.1 0.1 1"),
        ("y", [0, 0.025, 0], "0.1 0.9 0.1 1"),
        ("z", [0, 0, 0.035], "0.1 0.3 1 1"),
    ]:
        ET.SubElement(
            optical,
            "site",
            name=f"neon_camera_{axis}",
            type="capsule",
            fromto="0 0 0 " + values(end),
            size="0.00045",
            rgba=colour,
            group="5",
        )
    write_xml(root, asset_dir / "neon.xml")

    scene = ET.Element("mujoco", model="neon_inspection_scene")
    ET.SubElement(scene, "include", file="neon.xml")
    ET.SubElement(scene, "statistic", center="0 0.04 0.04", extent="0.22")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(
        visual,
        "global",
        offwidth="1600",
        offheight="1200",
        azimuth="70",
        elevation="-20",
    )
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.4 0.4 0.4",
        diffuse="0.7 0.7 0.7",
        specular="0.2 0.2 0.2",
    )
    ET.SubElement(visual, "map", znear="0.0005", zfar="20")
    ET.SubElement(visual, "rgba", haze="0.12 0.15 0.2 1")
    asset = ET.SubElement(scene, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="neon_floor",
        type="2d",
        builtin="checker",
        rgb1="0.24 0.28 0.34",
        rgb2="0.17 0.20 0.25",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="neon_floor",
        texture="neon_floor",
        texrepeat="12 12",
        reflectance="0.1",
    )
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 0.8", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.08",
        size="1 1 0.01",
        material="neon_floor",
        contype="0",
        conaffinity="0",
    )
    # Objects in front of the scene camera make its view easy to inspect.
    for x, colour in [
        (-0.09, "0.8 0.12 0.1 1"),
        (0, "0.1 0.7 0.2 1"),
        (0.09, "0.1 0.25 0.9 1"),
    ]:
        ET.SubElement(
            world,
            "geom",
            name=f"target_{x}",
            type="box",
            pos=values([x, -0.4, -0.03]),
            size="0.025 0.025 0.025",
            rgba=colour,
            contype="0",
            conaffinity="0",
        )
    write_xml(scene, asset_dir / "scene.xml")
    return asset_dir / "scene.xml"
