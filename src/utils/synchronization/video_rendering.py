"""Reusable synchronization video rendering helpers."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from utils.synchronization.clocks import utc_iso
from utils.synchronization.neon_video import (
    read_video_frame,
)
from utils.synchronization.projection import (
    bone_color,
    make_projection,
    nearest_index,
    project_to_pixels,
    projected_coordinates,
    quaternion_xyzw_to_matrix,
    unit_scale_to_meters,
)
from utils.synchronization.video_models import (
    MotiveMotion,
    NeonVideo,
    Projection,
)
from utils.synchronization.video_settings import (
    NS_PER_SECOND,
)


def render_rigid_body_panel(
    motion: MotiveMotion,
    frame_index: int,
    projection: Projection,
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), (24, 27, 32), dtype=np.uint8)
    for x in range(0, width, max(40, width // 12)):
        cv2.line(panel, (x, 142), (x, height - 42), (36, 40, 47), 1)
    for y in range(150, height - 40, max(40, height // 10)):
        cv2.line(panel, (0, y), (width, y), (36, 40, 47), 1)

    positions = motion.positions[:, 0]
    origin = positions[frame_index]
    valid_origin = np.isfinite(origin).all()
    trail_start = max(0, frame_index - round(5.0 * motion.frame_rate_hz))
    trail = positions[trail_start : frame_index + 1]
    trail_valid = np.isfinite(trail).all(axis=1)
    if np.count_nonzero(trail_valid) >= 2:
        trail_pixels = project_to_pixels(trail[trail_valid], projection, width, height)
        cv2.polylines(
            panel,
            [np.rint(trail_pixels).astype(np.int32)],
            False,
            (145, 95, 220),
            2,
            cv2.LINE_AA,
        )

    rotation = None
    if motion.quaternions_xyzw is not None:
        rotation = quaternion_xyzw_to_matrix(motion.quaternions_xyzw[frame_index])
    if valid_origin and rotation is not None:
        meters_per_unit = unit_scale_to_meters(motion.source_units)
        axis_length = 0.18 / meters_per_unit
        half_box = 0.045 / meters_per_unit
        local_corners = np.array(
            [
                [x, y, z]
                for x in (-half_box, half_box)
                for y in (-half_box, half_box)
                for z in (-half_box, half_box)
            ],
            dtype=np.float64,
        )
        world_corners = local_corners @ rotation.T + origin
        corner_pixels = np.rint(
            project_to_pixels(world_corners, projection, width, height)
        ).astype(int)
        cube_edges = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )
        for left, right in cube_edges:
            cv2.line(
                panel,
                tuple(corner_pixels[left]),
                tuple(corner_pixels[right]),
                (215, 210, 120),
                2,
                cv2.LINE_AA,
            )

        axis_endpoints = origin[None, :] + rotation.T * axis_length
        axis_pixels = np.rint(
            project_to_pixels(axis_endpoints, projection, width, height)
        ).astype(int)
        origin_pixel = tuple(
            np.rint(
                project_to_pixels(origin[None, :], projection, width, height)[0]
            ).astype(int)
        )
        axis_styles = (
            ("X", (70, 80, 255)),
            ("Y", (70, 240, 90)),
            ("Z", (255, 150, 60)),
        )
        for endpoint, (label, color) in zip(axis_pixels, axis_styles):
            endpoint_tuple = tuple(endpoint)
            cv2.arrowedLine(
                panel,
                origin_pixel,
                endpoint_tuple,
                color,
                4,
                cv2.LINE_AA,
                tipLength=0.16,
            )
            put_text(
                panel,
                label,
                (endpoint_tuple[0] + 5, endpoint_tuple[1] - 5),
                0.55,
                color,
                2,
            )
        cv2.circle(panel, origin_pixel, 7, (255, 255, 255), -1, cv2.LINE_AA)
        put_text(
            panel,
            (
                f"Position: [{origin[0]:+.3f}, {origin[1]:+.3f}, "
                f"{origin[2]:+.3f}] {motion.source_units}"
            ),
            (18, 166),
            0.46,
            (210, 215, 225),
        )
    else:
        put_text(
            panel,
            "Rigid body tracking missing in this frame",
            (18, 175),
            0.58,
            (80, 120, 255),
            2,
        )

    cv2.rectangle(panel, (0, 82), (width, 140), (14, 16, 20), -1)
    put_text(panel, "OptiTrack rigid body", (18, 110), 0.72, (90, 220, 255), 2)
    put_text(
        panel,
        f"Object: {motion.object_name} | axes X/Y/Z | trail: last 5 s",
        (18, 133),
        0.43,
        (190, 195, 205),
    )
    return panel


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_motive_panel(
    motion: MotiveMotion,
    frame_index: int,
    projection: Projection,
    width: int,
    height: int,
) -> np.ndarray:
    if motion.visualization_mode == "rigid-body":
        return render_rigid_body_panel(motion, frame_index, projection, width, height)
    panel = np.full((height, width, 3), (24, 27, 32), dtype=np.uint8)
    for x in range(0, width, max(40, width // 12)):
        cv2.line(panel, (x, 142), (x, height - 42), (36, 40, 47), 1)
    for y in range(150, height - 40, max(40, height // 10)):
        cv2.line(panel, (0, y), (width, y), (36, 40, 47), 1)

    points = projected_coordinates(motion.positions[frame_index])
    valid = np.isfinite(points).all(axis=1)
    center = np.array([width / 2.0, height * 0.60])
    pixels = np.empty_like(points)
    pixels[:, 0] = center[0] + (points[:, 0] - projection.center[0]) * projection.scale
    pixels[:, 1] = center[1] - (points[:, 1] - projection.center[1]) * projection.scale

    for parent, child in motion.edges:
        if not (valid[parent] and valid[child]):
            continue
        parent_pixel = tuple(np.rint(pixels[parent]).astype(int))
        child_pixel = tuple(np.rint(pixels[child]).astype(int))
        local_child = motion.bone_names[child].rpartition(":")[-1]
        cv2.line(
            panel,
            parent_pixel,
            child_pixel,
            bone_color(local_child),
            3,
            cv2.LINE_AA,
        )

    label_names = {"Head", "LHand", "RHand", "LFoot", "RFoot"}
    for index, (name, is_valid) in enumerate(zip(motion.bone_names, valid)):
        if not is_valid:
            continue
        local_name = name.rpartition(":")[-1]
        pixel = tuple(np.rint(pixels[index]).astype(int))
        cv2.circle(panel, pixel, 5, bone_color(local_name), -1, cv2.LINE_AA)
        if local_name in label_names:
            put_text(
                panel,
                local_name,
                (pixel[0] + 7, pixel[1] - 6),
                scale=0.40,
                color=(220, 225, 230),
            )

    cv2.rectangle(panel, (0, 82), (width, 140), (14, 16, 20), -1)
    put_text(panel, "OptiTrack Motive", (18, 110), 0.72, (90, 220, 255), 2)
    put_text(
        panel,
        f"Skeleton: {motion.skeleton_namespace}",
        (18, 133),
        0.43,
        (190, 195, 205),
    )
    return panel


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def overlay_timing(
    canvas: np.ndarray,
    ubuntu_ns: int,
    local_timezone: ZoneInfo,
    neon_index: int,
    neon_delta_ms: float,
    motive_frame: int,
    motive_delta_ms: float,
) -> None:
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 82), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 42), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    local_value = dt.datetime.fromtimestamp(
        ubuntu_ns / NS_PER_SECOND, tz=local_timezone
    )
    put_text(
        canvas,
        f"Ubuntu Unix ns: {ubuntu_ns}",
        (20, 29),
        0.68,
        (255, 255, 255),
        2,
    )
    put_text(
        canvas,
        f"Ubuntu UTC: {utc_iso(ubuntu_ns)}",
        (20, 57),
        0.58,
        (225, 235, 245),
        1,
    )
    put_text(
        canvas,
        f"Ubuntu local: {local_value.isoformat(timespec='milliseconds')}",
        (width // 2 + 10, 57),
        0.52,
        (225, 235, 245),
        1,
    )
    put_text(
        canvas,
        f"Neon frame {neon_index}  dt={neon_delta_ms:+.2f} ms",
        (18, height - 14),
        0.52,
        (100, 230, 255),
    )
    put_text(
        canvas,
        f"Motive frame {motive_frame}  dt={motive_delta_ms:+.2f} ms",
        (width // 2 + 18, height - 14),
        0.52,
        (100, 230, 255),
    )


def render_video(
    motive: MotiveMotion,
    neon: NeonVideo,
    output_path: Path,
    output_width: int,
    output_height: int,
    output_fps: float,
    codec: str,
    max_duration: float | None,
    local_timezone: ZoneInfo,
) -> dict[str, object]:
    overlap_start = max(int(motive.ubuntu_ns[0]), int(neon.ubuntu_ns[0]))
    overlap_end = min(int(motive.ubuntu_ns[-1]), int(neon.ubuntu_ns[-1]))
    if overlap_end <= overlap_start:
        raise ValueError(
            "Neon 与 Motive 在 Ubuntu 时间轴上没有重叠：\n"
            f"  Neon : {utc_iso(int(neon.ubuntu_ns[0]))} .. "
            f"{utc_iso(int(neon.ubuntu_ns[-1]))}\n"
            f"  Motive: {utc_iso(int(motive.ubuntu_ns[0]))} .. "
            f"{utc_iso(int(motive.ubuntu_ns[-1]))}\n"
            "请检查是否使用同一次录制、Motive 时区，以及 "
            "Windows→Ubuntu offset。"
        )
    if max_duration is not None:
        overlap_end = min(
            overlap_end,
            overlap_start + round(max_duration * NS_PER_SECOND),
        )
    duration_s = (overlap_end - overlap_start) / NS_PER_SECOND
    frame_count = max(1, int(math.floor(duration_s * output_fps)))
    output_times = overlap_start + np.rint(
        np.arange(frame_count, dtype=np.float64) / output_fps * NS_PER_SECOND
    ).astype(np.int64)

    if output_width < 640 or output_height < 360:
        raise ValueError("输出分辨率至少应为 640x360")
    if len(codec) != 4:
        raise ValueError("--codec 必须是四字符代码，例如 mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise ValueError(f"无法创建 MP4 {output_path}；可尝试 --codec mp4v 或 avc1")

    capture = cv2.VideoCapture(str(neon.video_path))
    if not capture.isOpened():
        writer.release()
        raise ValueError(f"无法重新打开 Neon 视频：{neon.video_path}")
    half_width = output_width // 2
    right_width = output_width - half_width
    projection = make_projection(motive, right_width, output_height)
    video_state: dict[str, object] = {}
    print(
        f"正在生成 {frame_count} 帧，{output_fps:g} FPS，"
        f"时长 {duration_s:.3f} 秒 ..."
    )
    try:
        for output_index, timestamp_ns in enumerate(output_times):
            timestamp = int(timestamp_ns)
            neon_index = nearest_index(neon.ubuntu_ns, timestamp)
            motive_index = nearest_index(motive.ubuntu_ns, timestamp)
            neon_frame = read_video_frame(capture, neon_index, video_state)
            left_panel = letterbox(neon_frame, half_width, output_height)
            cv2.rectangle(left_panel, (0, 82), (half_width, 140), (14, 16, 20), -1)
            put_text(
                left_panel,
                "Pupil Labs Neon scene video",
                (18, 118),
                0.70,
                (90, 220, 255),
                2,
            )
            right_panel = render_motive_panel(
                motive, motive_index, projection, right_width, output_height
            )
            canvas = np.concatenate((left_panel, right_panel), axis=1)
            neon_delta_ms = (int(neon.ubuntu_ns[neon_index]) - timestamp) / 1_000_000
            motive_delta_ms = (
                int(motive.ubuntu_ns[motive_index]) - timestamp
            ) / 1_000_000
            overlay_timing(
                canvas,
                timestamp,
                local_timezone,
                neon_index,
                neon_delta_ms,
                int(motive.frame_ids[motive_index]),
                motive_delta_ms,
            )
            writer.write(canvas)
            if output_index and output_index % max(1, round(output_fps * 10)) == 0:
                print(
                    f"  {output_index}/{frame_count} "
                    f"({100.0 * output_index / frame_count:.1f}%)"
                )
    finally:
        capture.release()
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError("MP4 编码结束但输出文件为空")
    return {
        "overlap_start_ubuntu_ns": overlap_start,
        "overlap_end_ubuntu_ns": int(output_times[-1]),
        "overlap_start_utc": utc_iso(overlap_start),
        "overlap_end_utc": utc_iso(int(output_times[-1])),
        "duration_seconds": duration_s,
        "output_frame_count": frame_count,
        "output_fps": output_fps,
        "output_width": output_width,
        "output_height": output_height,
        "codec": codec,
    }
