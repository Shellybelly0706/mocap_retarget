"""Reusable synchronization neon video helpers."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.synchronization.clocks import read_neon_time_file
from utils.synchronization.video_models import (
    NeonVideo,
)
from utils.synchronization.video_settings import (
    NS_PER_SECOND,
)


def find_neon_file(
    recording_dir: Path, requested: Path | None, default_name: str
) -> Path:
    if requested is not None:
        return requested
    candidate = recording_dir / default_name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Neon 录制目录中没有 {default_name!r}；请通过参数显式指定"
        )
    return candidate


def load_neon_video(
    recording_dir: Path,
    video_path: Path | None,
    time_path: Path | None,
    companion_to_ubuntu,
) -> NeonVideo:
    info_path = recording_dir / "info.json"
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    try:
        recording_start_companion_ns = int(info["start_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Neon info.json 缺少有效 start_time") from exc
    recording_start_ubuntu_ns = companion_to_ubuntu(recording_start_companion_ns)
    resolved_video = find_neon_file(
        recording_dir, video_path, "Neon Scene Camera v1 ps1.mp4"
    )
    resolved_time = time_path
    if resolved_time is None:
        inferred_time = resolved_video.with_suffix(".time")
        if inferred_time.is_file():
            resolved_time = inferred_time

    capture = cv2.VideoCapture(str(resolved_video))
    if not capture.isOpened():
        raise ValueError(f"无法打开 Neon 视频：{resolved_video}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Neon 视频 FPS 或分辨率无效")

    if resolved_time is not None:
        companion = np.asarray(read_neon_time_file(resolved_time), dtype=np.int64)
        if video_frame_count > 0:
            usable_count = min(video_frame_count, len(companion))
            if usable_count == 0:
                raise ValueError("Neon 视频或 .time 文件没有帧")
            if abs(video_frame_count - len(companion)) > 2:
                print(
                    "警告：Neon 视频帧数与 .time 数量不同，将使用较短长度："
                    f"video={video_frame_count}, time={len(companion)}",
                    file=sys.stderr,
                )
            companion = companion[:usable_count]
    else:
        if video_frame_count <= 0:
            raise ValueError("视频没有 .time 文件且无法读取视频帧数")
        print(
            "警告：没有 Neon scene .time 文件；将假定视频第 0 帧就是 "
            "info.json start_time，可能引入启动延迟误差。",
            file=sys.stderr,
        )
        companion = recording_start_companion_ns + np.rint(
            np.arange(video_frame_count, dtype=np.float64) / fps * NS_PER_SECOND
        ).astype(np.int64)
    ubuntu = np.asarray(
        [companion_to_ubuntu(int(value)) for value in companion],
        dtype=np.int64,
    )
    if np.any(np.diff(ubuntu) <= 0):
        raise ValueError("转换后的 Neon 视频 Ubuntu 时间戳不是严格递增")

    template = info.get("template_data")
    recording_name = recording_dir.name
    if isinstance(template, dict) and isinstance(template.get("data"), dict):
        recording_name = str(template.get("recording_name") or recording_name)
    return NeonVideo(
        recording_dir=recording_dir,
        video_path=resolved_video,
        time_path=resolved_time,
        recording_start_companion_ns=recording_start_companion_ns,
        recording_start_ubuntu_ns=recording_start_ubuntu_ns,
        companion_ns=companion,
        ubuntu_ns=ubuntu,
        fps=fps,
        width=width,
        height=height,
        recording_name=recording_name,
    )


def read_video_frame(
    capture: cv2.VideoCapture,
    target_index: int,
    state: dict[str, object],
) -> np.ndarray:
    current_index = int(state.get("index", -1))
    current_frame = state.get("frame")
    if target_index < current_index or target_index - current_index > 120:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"无法读取 Neon 视频第 {target_index} 帧")
        state["index"] = target_index
        state["frame"] = frame
        return frame
    while current_index < target_index:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"无法读取 Neon 视频第 {current_index + 1} 帧")
        current_index += 1
        current_frame = frame
    if current_frame is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, current_frame = capture.read()
        if not ok:
            raise ValueError(f"无法读取 Neon 视频第 {target_index} 帧")
        current_index = target_index
    state["index"] = current_index
    state["frame"] = current_frame
    return np.asarray(current_frame)
