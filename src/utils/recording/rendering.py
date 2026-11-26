"""Render a paired inspection frame without importing the desktop GUI."""

from pathlib import Path

import cv2
import mujoco
from PIL import Image, ImageDraw, ImageFont

from utils.synchronization.neon_video import read_video_frame


def paired_image(scene, renderer, capture, state, frame):
    scene.set_frame(frame)
    renderer.update_scene(scene.data, camera=scene.camera, scene_option=scene.options)
    left = Image.fromarray(renderer.render())
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
    draw.text((12, 8), scene.label, font=font, fill="white")
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
