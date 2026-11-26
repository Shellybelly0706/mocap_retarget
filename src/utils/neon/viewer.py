"""Validate, render, and inspect the shipped Neon MuJoCo model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from utils.paths import ASSETS_DIR

from .camera import load_scene_camera_calibration
from .imu import camera_to_imu

NEON_ASSET_DIR = ASSETS_DIR / "neon"


def load_neon_model(asset_dir: Path = NEON_ASSET_DIR):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str((Path(asset_dir) / "scene.xml").resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def validate_neon_model(asset_dir: Path = NEON_ASSET_DIR) -> dict:
    """Check mesh scale, camera axes, and the compiled pixel intrinsics."""
    import mujoco

    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir / "source.json").read_text())
    calibration = load_scene_camera_calibration(
        asset_dir / manifest["calibration"]["file"]
    )
    model, data = load_neon_model(asset_dir)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "neon_scene_camera")
    optical = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "neon_scene_camera_optical"
    )
    if camera < 0 or optical < 0:
        raise ValueError("Neon optical frame or scene camera is missing")
    rotation = data.xmat[optical].reshape(3, 3)
    imu = model.body("neon_imu").id
    rotation_i_c, translation_i_c = camera_to_imu()
    rotation_w_i = data.xmat[imu].reshape(3, 3)
    np.testing.assert_allclose(rotation_w_i.T @ rotation, rotation_i_c, atol=1e-10)
    np.testing.assert_allclose(
        rotation_w_i.T @ (data.xpos[optical] - data.xpos[imu]),
        translation_i_c,
        atol=1e-10,
    )
    camera_rotation = data.cam_xmat[camera].reshape(3, 3)
    np.testing.assert_allclose(data.cam_xpos[camera], data.xpos[optical], atol=1e-10)
    np.testing.assert_allclose(
        camera_rotation, rotation @ np.diag([1, -1, -1]), atol=1e-10
    )

    width, height = calibration.resolution
    scaled = model.cam_intrinsic[camera] / np.tile(model.cam_sensorsize[camera], 2)
    actual = np.array(
        [
            [scaled[0] * width, 0, width / 2 - scaled[2] * width],
            [0, scaled[1] * height, height / 2 - scaled[3] * height],
            [0, 0, 1],
        ]
    )
    np.testing.assert_allclose(actual, calibration.matrix, rtol=1e-6, atol=1e-4)
    if tuple(model.cam_resolution[camera]) != calibration.resolution:
        raise ValueError("Camera resolution does not match calibration")

    points = []
    for geom in range(model.ngeom):
        mesh = model.geom_dataid[geom]
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        address, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[address : address + count]
        points.append(
            vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
        if model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0:
            raise ValueError("The reference CAD must remain visual-only")
    dimensions = np.ptp(np.concatenate(points), axis=0)
    if not (
        0.14 < dimensions[0] < 0.18
        and 0.14 < dimensions[1] < 0.19
        and 0.04 < dimensions[2] < 0.08
    ):
        raise ValueError(f"Unexpected glasses dimensions in metres: {dimensions}")
    return {
        "serial": calibration.serial,
        "mesh_count": model.nmesh,
        "triangles": int(model.mesh_facenum.sum()),
        "dimensions_m": dimensions.tolist(),
        "resolution": list(calibration.resolution),
        "camera_matrix": actual.tolist(),
        "camera_position_world_m": data.cam_xpos[camera].tolist(),
        "pose_source": "CAD lens-opening centre; not calibrated to the optical centre",
        "imu_position_cad_m": model.body_pos[imu].tolist(),
        "imu_reference": "Module top bar; manufacturer nominal camera-to-IMU transform",
    }


def inspection_camera(model):
    import mujoco

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0.04, 0.035]
    camera.distance = 0.34
    camera.azimuth = 70
    camera.elevation = -18
    return camera


def render_neon(
    output: Path, *, camera: str = "orbit", asset_dir: Path = NEON_ASSET_DIR
) -> Path:
    """Save an orbit preview or an undistorted pinhole scene-camera image."""
    import mujoco
    from PIL import Image

    model, data = load_neon_model(asset_dir)
    scene_view = camera == "scene"
    width, height = (1600, 1200) if scene_view else (1200, 900)
    option = mujoco.MjvOption()
    option.sitegroup[5] = not scene_view
    option.sitegroup[4] = not scene_view
    selected = "neon_scene_camera" if scene_view else inspection_camera(model)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=selected, scene_option=option)
        pixels = renderer.render().copy()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output)
    return output


def view_neon(*, camera: str = "orbit", asset_dir: Path = NEON_ASSET_DIR) -> None:
    import mujoco
    import mujoco.viewer

    model, data = load_neon_model(asset_dir)
    state = {"scene": camera == "scene", "frames": True}

    def key_callback(key):
        if key in (ord("C"), ord("c")):
            state["scene"] = not state["scene"]
        elif key in (ord("F"), ord("f")):
            state["frames"] = not state["frames"]

    print("C: switch orbit/scene camera | F: toggle camera/IMU frames | Esc: close")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        orbit = inspection_camera(model)
        viewer.cam.lookat[:] = orbit.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
            orbit.distance,
            orbit.azimuth,
            orbit.elevation,
        )
        while viewer.is_running():
            with viewer.lock():
                viewer.cam.type = (
                    mujoco.mjtCamera.mjCAMERA_FIXED
                    if state["scene"]
                    else mujoco.mjtCamera.mjCAMERA_FREE
                )
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, "neon_scene_camera"
                )
                viewer.opt.sitegroup[5] = state["frames"] and not state["scene"]
                viewer.opt.sitegroup[4] = state["frames"] and not state["scene"]
            viewer.sync()
            time.sleep(1 / 60)
