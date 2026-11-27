"""Render a paired inspection frame without importing the desktop GUI."""

from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.synchronization.neon_video import read_video_frame


def render_mujoco(scene, renderer) -> np.ndarray:
    """Render the scene, optionally compositing an internal skeleton on top.

    MuJoCo skins write depth even when translucent, so a real joint hierarchy
    inside the body would otherwise disappear.  The x-ray pass keeps the exact
    3-D joints and current camera, renders them with only the skin hidden, and
    composites just the pixels changed by skeleton group 2 over the normal
    image.
    """
    renderer.update_scene(
        scene.data, camera=scene.camera, scene_option=scene.options
    )
    base = renderer.render().copy()
    if (
        not scene.skeleton_xray
        or not scene.options.geomgroup[2]
        or not scene.options.skingroup[4]
    ):
        return base

    skin_visible = int(scene.options.skingroup[4])
    skeleton_visible = int(scene.options.geomgroup[2])
    try:
        scene.options.skingroup[4] = 0
        renderer.update_scene(
            scene.data, camera=scene.camera, scene_option=scene.options
        )
        skeleton = renderer.render().copy()
        scene.options.geomgroup[2] = 0
        renderer.update_scene(
            scene.data, camera=scene.camera, scene_option=scene.options
        )
        without_skeleton = renderer.render().copy()
    finally:
        scene.options.skingroup[4] = skin_visible
        scene.options.geomgroup[2] = skeleton_visible
        renderer.update_scene(
            scene.data, camera=scene.camera, scene_option=scene.options
        )

    changed = np.max(
        np.abs(skeleton.astype(np.int16) - without_skeleton.astype(np.int16)),
        axis=2,
    )
    mask = changed > 2
    output = base.copy()
    output[mask] = skeleton[mask]
    return output


def paired_image(scene, renderer, capture, state, frame):
    scene.set_frame(frame)
    left = Image.fromarray(render_mujoco(scene, renderer))
    right = Image.fromarray(
        cv2.cvtColor(read_video_frame(capture, frame, state), cv2.COLOR_BGR2RGB)
    )
    right = right.resize(left.size, Image.Resampling.LANCZOS)
    image = Image.new("RGB", (left.width * 2, left.height + 100), "#121923")
    image.paste(left, (0, 55))
    image.paste(right, (left.width, 55))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    row = scene.session.mapping[frame]
    draw.text(
        (12, 8),
        f"{scene.label} | {scene.skeleton_label}",
        font=font,
        fill="white",
    )
    draw.text(
        (left.width + 12, 8),
        "Neon front camera | same aligned frame",
        font=font,
        fill="white",
    )
    draw.text(
        (12, 30),
        (
            "Glasses: IMU orientation, geometric nose position"
            if scene.use_imu
            else "Glasses: provisional visual mount"
        )
        + " | RGB axes: camera + IMU",
        font=font,
        fill="#ffd28c",
    )
    draw.text(
        (12, left.height + 65),
        f"Frame {frame}/{len(scene.session.times)-1}   t={scene.session.times[frame]:.3f}s   source Neon={row['neon_source_frame']}  Motive={row['mocap_source_frame']}   delta={float(row['mocap_delta_ms']):+.3f}ms",
        font=font,
        fill="white",
    )
    return image


def render_preview(scene, output: Path, frame: int = 0):
    if not 0 <= frame < len(scene.session.times):
        raise ValueError("Preview frame is outside the aligned recording")
    capture = cv2.VideoCapture(
        str(scene.session.directory / "front_camera_aligned.mp4")
    )
    try:
        with mujoco.Renderer(scene.model, width=640, height=480) as renderer:
            image = paired_image(scene, renderer, capture, {}, frame)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
    finally:
        capture.release()
