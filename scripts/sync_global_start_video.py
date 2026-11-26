#!/usr/bin/env python3
"""Create a Neon/Motive split-screen video using only global timestamps.

No motion feature or manually selected event is used.  The time domains are:

* Neon Companion -> Ubuntu: optionally interpolated from Pupil Labs Time Echo
  records.  Without ``--offset-log``, Companion timestamps are used directly.
* Motive/Windows -> Ubuntu: Motive ``Capture Start Time`` interpreted in the
  requested timezone, plus an optional measured ``Ubuntu - Windows`` offset.

Important limitation
--------------------
Time Echo measures Ubuntu minus *Neon Companion*.  It says nothing about the
Windows clock running Motive.  Therefore the default ``--motive-ubuntu-offset-ms
0`` assumes the Windows wall clock and Ubuntu wall clock were already
synchronized.  If they were not, global-start-only alignment inherits that
unknown error.  Use ``--motive-ubuntu-offset-ms`` when that offset is known.

Time Echo correction is optional because Neon timestamps are already Unix
timestamps and the clock difference can be negligible for some workflows.
Omitting it explicitly assumes ``Ubuntu time == Neon Companion time``.

The generated MP4 contains no audio.  Its left panel is the Neon scene video,
the right panel is an isometric Motive skeleton visualization, and every frame
is labelled with its Ubuntu Unix/UTC/local time.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np

try:
    from sync_motive_neon import (
        make_clock_converter,
        read_clock_anchors,
        read_neon_time_file,
        utc_iso,
    )
except ImportError:
    try:
        from scripts.sync_motive_neon import (
            make_clock_converter,
            read_clock_anchors,
            read_neon_time_file,
            utc_iso,
        )
    except ImportError as exc:  # pragma: no cover - unusual invocation only
        raise RuntimeError(
            "无法导入同目录的 sync_motive_neon.py"
        ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sync"
NS_PER_SECOND = 1_000_000_000


SKELETON_PARENTS: dict[str, str | None] = {
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
for _side in ("L", "R"):
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        SKELETON_PARENTS[f"{_side}{_finger}1"] = f"{_side}Hand"
        SKELETON_PARENTS[f"{_side}{_finger}2"] = f"{_side}{_finger}1"
        SKELETON_PARENTS[f"{_side}{_finger}3"] = f"{_side}{_finger}2"


@dataclass(frozen=True)
class MotiveMotion:
    path: Path
    take_name: str
    capture_start_text: str
    capture_start_ubuntu_ns: int
    frame_rate_hz: float
    frame_ids: np.ndarray
    ubuntu_ns: np.ndarray
    visualization_mode: str
    object_name: str
    skeleton_namespace: str
    bone_names: tuple[str, ...]
    positions: np.ndarray
    quaternions_xyzw: np.ndarray | None
    edges: tuple[tuple[int, int], ...]
    source_units: str


@dataclass(frozen=True)
class NeonVideo:
    recording_dir: Path
    video_path: Path
    time_path: Path | None
    recording_start_companion_ns: int
    recording_start_ubuntu_ns: int
    companion_ns: np.ndarray
    ubuntu_ns: np.ndarray
    fps: float
    width: int
    height: int
    recording_name: str


@dataclass(frozen=True)
class Projection:
    center: np.ndarray
    scale: float


def normalized(value: str) -> str:
    return value.strip().lower()


def padded_cell(row: Sequence[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def metadata_pairs(rows: Sequence[Sequence[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        for index in range(0, len(row) - 1, 2):
            key = row[index].strip()
            if key:
                result[normalized(key)] = row[index + 1].strip()
    return result


def find_header_row(rows: Sequence[Sequence[str]], label: str) -> int:
    wanted = normalized(label)
    matches = [
        index
        for index, row in enumerate(rows)
        if any(normalized(cell) == wanted for cell in row[:2])
    ]
    if not matches:
        raise ValueError(f"Motive CSV 缺少 {label!r} 表头行")
    return matches[-1]


def datetime_to_unix_ns(value: dt.datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("datetime 必须带时区")
    utc = value.astimezone(dt.timezone.utc)
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta = utc - epoch
    return (
        (delta.days * 86400 + delta.seconds) * NS_PER_SECOND
        + delta.microseconds * 1000
    )


def parse_motive_start_time(text: str, timezone_name: str) -> int:
    formats = (
        "%Y-%m-%d %I.%M.%S.%f %p",
        "%Y-%m-%d %I:%M:%S.%f %p",
        "%Y-%m-%d %H.%M.%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %I.%M.%S %p",
        "%Y-%m-%d %H:%M:%S",
    )
    parsed: dt.datetime | None = None
    for date_format in formats:
        try:
            parsed = dt.datetime.strptime(text.strip(), date_format)
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError(
            f"无法解析 Motive Capture Start Time {text!r}；"
            "支持 Motive 常见的带毫秒 AM/PM 格式"
        )
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"系统中不存在时区 {timezone_name!r}") from exc
    return datetime_to_unix_ns(parsed.replace(tzinfo=timezone))


def discover_motive_header(stream: csv.reader) -> tuple[list[list[str]], list[str]]:
    header: list[list[str]] = []
    for row in stream:
        header.append(row)
        if (
            normalized(padded_cell(row, 0)) == "frame"
            and normalized(padded_cell(row, 1)) in {"time", "time (seconds)"}
        ):
            return header, row
    raise ValueError("Motive CSV 中没有 Frame/Time 数据表头")


def select_skeleton_namespace(
    complete_bones: dict[str, dict[str, int]], requested: str | None
) -> str:
    namespaces: dict[str, int] = {}
    for name in complete_bones:
        namespace, separator, _ = name.rpartition(":")
        if separator:
            namespaces[namespace] = namespaces.get(namespace, 0) + 1
    if not namespaces:
        raise ValueError("Motive CSV 中没有带命名空间的完整 Bone Position")
    if requested is not None:
        exact = [name for name in namespaces if name == requested]
        if not exact:
            exact = [name for name in namespaces if normalized(name) == normalized(requested)]
        if len(exact) != 1:
            raise ValueError(
                f"找不到唯一 skeleton {requested!r}；可用值："
                + ", ".join(sorted(namespaces))
            )
        return exact[0]
    return max(sorted(namespaces), key=lambda name: namespaces[name])


def build_skeleton_edges(
    namespace: str, bone_names: Sequence[str]
) -> tuple[tuple[int, int], ...]:
    indices = {name: index for index, name in enumerate(bone_names)}
    root = f"{namespace}:{namespace}"
    edges: list[tuple[int, int]] = []
    for child_name, child_index in indices.items():
        _, _, local_name = child_name.rpartition(":")
        parent_local = SKELETON_PARENTS.get(local_name)
        if local_name in {"Ab", "LThigh", "RThigh"}:
            parent_name = root
        elif parent_local is not None:
            parent_name = f"{namespace}:{parent_local}"
        else:
            continue
        if (
            parent_name not in indices
            and parent_local in {"LHand", "RHand"}
            and local_name.endswith("1")
        ):
            parent_name = root
        if parent_name in indices and parent_name != child_name:
            edges.append((indices[parent_name], child_index))
    return tuple(edges)


def load_motive_motion(
    path: Path,
    timezone_name: str,
    ubuntu_minus_windows_ms: float,
    visualization_mode: str,
    requested_rigid_body: str | None,
    requested_skeleton: str | None,
) -> MotiveMotion:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header, axis_row_values = discover_motive_header(reader)
        type_row = header[find_header_row(header, "Type")]
        name_row = header[find_header_row(header, "Name")]
        quantity_candidates = [
            row
            for row in header
            if any(normalized(cell) == "position" for cell in row)
        ]
        if not quantity_candidates:
            raise ValueError("Motive CSV 缺少 Position 表头")
        quantity_row = quantity_candidates[-1]
        metadata = metadata_pairs(header[:-1])

        rate_text = metadata.get("export frame rate") or metadata.get(
            "capture frame rate"
        )
        start_text = metadata.get("capture start time")
        if not rate_text or not start_text:
            raise ValueError(
                "Motive CSV 必须包含 Export Frame Rate 和 Capture Start Time"
            )
        frame_rate = float(rate_text)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError(f"无效 Motive 帧率：{rate_text!r}")
        start_ns = parse_motive_start_time(start_text, timezone_name) + round(
            ubuntu_minus_windows_ms * 1_000_000
        )

        width = max(
            len(type_row),
            len(name_row),
            len(quantity_row),
            len(axis_row_values),
        )
        position_axes: dict[tuple[str, str], dict[str, int]] = {}
        rotation_axes: dict[tuple[str, str], dict[str, int]] = {}
        for column in range(width):
            object_type = padded_cell(type_row, column)
            name = padded_cell(name_row, column)
            quantity = normalized(padded_cell(quantity_row, column))
            axis = padded_cell(axis_row_values, column).upper()
            if not object_type or not name:
                continue
            key = (object_type, name)
            if quantity == "position" and axis in {"X", "Y", "Z"}:
                position_axes.setdefault(key, {})[axis] = column
            elif quantity == "rotation" and axis in {"X", "Y", "Z", "W"}:
                rotation_axes.setdefault(key, {})[axis] = column

        complete_positions = {
            key: axes
            for key, axes in position_axes.items()
            if set(axes) == {"X", "Y", "Z"}
        }
        complete_rotations = {
            key: axes
            for key, axes in rotation_axes.items()
            if set(axes) == {"X", "Y", "Z", "W"}
        }
        rigid_names = sorted(
            name
            for object_type, name in complete_positions
            if object_type == "Rigid Body"
            and (object_type, name) in complete_rotations
        )

        quaternion_columns: tuple[int, int, int, int] | None = None
        namespace = ""
        if visualization_mode == "rigid-body":
            if requested_rigid_body is not None:
                matches = [name for name in rigid_names if name == requested_rigid_body]
                if not matches:
                    matches = [
                        name
                        for name in rigid_names
                        if normalized(name) == normalized(requested_rigid_body)
                    ]
                if len(matches) != 1:
                    raise ValueError(
                        f"找不到唯一刚体 {requested_rigid_body!r}；可用刚体："
                        + (", ".join(rigid_names) if rigid_names else "无")
                    )
                rigid_name = matches[0]
            elif len(rigid_names) == 1:
                rigid_name = rigid_names[0]
            elif not rigid_names:
                type_counts: dict[str, int] = {}
                for object_type, _ in complete_positions:
                    type_counts[object_type] = type_counts.get(object_type, 0) + 1
                summary = ", ".join(
                    f"{key}={value}" for key, value in sorted(type_counts.items())
                )
                raise ValueError(
                    "该 Motive CSV 没有导出任何 Rigid Body。"
                    f"当前完整 Position 类型为：{summary or '无'}。"
                    "请在 Motive 重新导出时启用 Rigid Bodies；"
                    "不要把 Skeleton 或 Unlabeled Marker 当成刚体姿态。"
                )
            else:
                raise ValueError(
                    "CSV 中有多个刚体，请使用 --rigid-body 指定："
                    + ", ".join(rigid_names)
                )
            key = ("Rigid Body", rigid_name)
            axes = complete_positions[key]
            rotation = complete_rotations[key]
            selected = [rigid_name]
            columns = [(axes["X"], axes["Y"], axes["Z"])]
            quaternion_columns = (
                rotation["X"],
                rotation["Y"],
                rotation["Z"],
                rotation["W"],
            )
            object_name = rigid_name
        else:
            complete_bones = {
                name: axes
                for (object_type, name), axes in complete_positions.items()
                if object_type == "Bone"
            }
            namespace = select_skeleton_namespace(
                complete_bones, requested_skeleton
            )
            selected = sorted(
                name
                for name in complete_bones
                if name.rpartition(":")[0] == namespace
            )
            columns = [
                (
                    complete_bones[name]["X"],
                    complete_bones[name]["Y"],
                    complete_bones[name]["Z"],
                )
                for name in selected
            ]
            object_name = namespace

        frames: list[int] = []
        samples: list[list[list[float]]] = []
        quaternion_samples: list[list[float]] = []
        for csv_row_number, row in enumerate(reader, len(header) + 1):
            frame_text = padded_cell(row, 0)
            if not frame_text:
                continue
            try:
                frames.append(int(frame_text))
                frame_positions: list[list[float]] = []
                for xyz_columns in columns:
                    xyz = []
                    for column in xyz_columns:
                        cell = padded_cell(row, column)
                        xyz.append(float(cell) if cell else math.nan)
                    frame_positions.append(xyz)
                samples.append(frame_positions)
                if quaternion_columns is not None:
                    quaternion = []
                    for column in quaternion_columns:
                        cell = padded_cell(row, column)
                        quaternion.append(float(cell) if cell else math.nan)
                    quaternion_samples.append(quaternion)
            except ValueError as exc:
                raise ValueError(
                    f"Motive CSV 第 {csv_row_number} 行含无效数字"
                ) from exc

    if not frames:
        raise ValueError("Motive CSV 中没有数据帧")
    frame_ids = np.asarray(frames, dtype=np.int64)
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Motive Frame 必须严格递增")
    positions = np.asarray(samples, dtype=np.float64)
    frame_ubuntu_ns = start_ns + np.rint(
        frame_ids.astype(np.float64) / frame_rate * NS_PER_SECOND
    ).astype(np.int64)
    bone_names = tuple(selected)
    quaternions = (
        np.asarray(quaternion_samples, dtype=np.float64)
        if quaternion_columns is not None
        else None
    )
    edges = (
        build_skeleton_edges(namespace, bone_names)
        if visualization_mode == "skeleton"
        else ()
    )
    if visualization_mode == "skeleton" and not edges:
        raise ValueError(f"无法为 skeleton {namespace!r} 建立骨骼连接")
    return MotiveMotion(
        path=path,
        take_name=metadata.get("take name", path.stem),
        capture_start_text=start_text,
        capture_start_ubuntu_ns=start_ns,
        frame_rate_hz=frame_rate,
        frame_ids=frame_ids,
        ubuntu_ns=frame_ubuntu_ns,
        visualization_mode=visualization_mode,
        object_name=object_name,
        skeleton_namespace=namespace,
        bone_names=bone_names,
        positions=positions,
        quaternions_xyzw=quaternions,
        edges=edges,
        source_units=metadata.get("length units", "unknown"),
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
    recording_start_ubuntu_ns = companion_to_ubuntu(
        recording_start_companion_ns
    )
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
            np.arange(video_frame_count, dtype=np.float64)
            / fps
            * NS_PER_SECOND
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


def nearest_index(sorted_values: np.ndarray, target: int) -> int:
    position = int(np.searchsorted(sorted_values, target, side="left"))
    if position <= 0:
        return 0
    if position >= len(sorted_values):
        return len(sorted_values) - 1
    before = int(sorted_values[position - 1])
    after = int(sorted_values[position])
    return position - 1 if target - before <= after - target else position


def projected_coordinates(positions: np.ndarray) -> np.ndarray:
    """Fixed isometric projection for Motive's usual Y-up coordinates."""

    result = np.empty((*positions.shape[:-1], 2), dtype=np.float64)
    result[..., 0] = positions[..., 0] - 0.38 * positions[..., 2]
    result[..., 1] = positions[..., 1] + 0.16 * positions[..., 2]
    return result


def make_projection(motion: MotiveMotion, width: int, height: int) -> Projection:
    projected = projected_coordinates(motion.positions)
    finite = np.isfinite(projected).all(axis=2)
    minimum_points = 2 if motion.visualization_mode == "rigid-body" else 10
    if np.count_nonzero(finite) < minimum_points:
        raise ValueError("Motive 有效位置不足，无法可视化")
    values = projected[finite]
    low = np.percentile(values, 1.0, axis=0)
    high = np.percentile(values, 99.0, axis=0)
    minimum_span = 0.5 if normalized(motion.source_units) in {"m", "meters"} else 500.0
    span = np.maximum(high - low, minimum_span)
    drawable_width = width * 0.82
    drawable_height = height * 0.62
    scale = min(drawable_width / span[0], drawable_height / span[1])
    return Projection(center=(low + high) / 2.0, scale=float(scale))


def bone_color(local_name: str) -> tuple[int, int, int]:
    if local_name.startswith("L"):
        return (255, 180, 55)
    if local_name.startswith("R"):
        return (70, 150, 255)
    return (80, 230, 130)


def project_to_pixels(
    world_points: np.ndarray,
    projection: Projection,
    width: int,
    height: int,
) -> np.ndarray:
    projected = projected_coordinates(np.asarray(world_points, dtype=np.float64))
    pixels = np.empty_like(projected)
    pixels[..., 0] = (
        width / 2.0
        + (projected[..., 0] - projection.center[0]) * projection.scale
    )
    pixels[..., 1] = (
        height * 0.60
        - (projected[..., 1] - projection.center[1]) * projection.scale
    )
    return pixels


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray | None:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        return None
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return None
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def unit_scale_to_meters(units: str) -> float:
    scales = {
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
    return scales.get(normalized(units), 1.0)


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
        trail_pixels = project_to_pixels(
            trail[trail_valid], projection, width, height
        )
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
        rotation = quaternion_xyzw_to_matrix(
            motion.quaternions_xyzw[frame_index]
        )
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
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
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
        return render_rigid_body_panel(
            motion, frame_index, projection, width, height
        )
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
        np.arange(frame_count, dtype=np.float64)
        / output_fps
        * NS_PER_SECOND
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
        raise ValueError(
            f"无法创建 MP4 {output_path}；可尝试 --codec mp4v 或 avc1"
        )

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
            cv2.rectangle(
                left_panel, (0, 82), (half_width, 140), (14, 16, 20), -1
            )
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
            neon_delta_ms = (
                int(neon.ubuntu_ns[neon_index]) - timestamp
            ) / 1_000_000
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "仅根据 Neon/Motive global 开始时间生成同步分屏 MP4；可选用 "
            "Time Echo 校正 Ubuntu 与 Neon Companion 的时钟误差"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motive-csv", type=Path, required=True)
    parser.add_argument("--neon-recording-dir", type=Path, required=True)
    parser.add_argument(
        "--offset-log",
        type=Path,
        help=(
            "可选：clock_offset_monitor.py 生成的 clock_offsets.jsonl；"
            "提供时校正 Ubuntu - Neon Companion 时钟误差，省略时假定两者相同"
        ),
    )
    parser.add_argument(
        "--neon-video",
        type=Path,
        help="默认使用录制目录中的 Neon Scene Camera v1 ps1.mp4",
    )
    parser.add_argument(
        "--neon-video-time",
        type=Path,
        help="默认使用与 scene MP4 同名的 .time",
    )
    parser.add_argument(
        "--motive-timezone",
        default="Europe/Berlin",
        help="解释 Motive Capture Start Time 所用的 Windows 时区",
    )
    parser.add_argument(
        "--motive-ubuntu-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Ubuntu - Motive Windows 的时钟 offset；0 表示假定两台电脑"
            "墙上时钟已同步"
        ),
    )
    parser.add_argument(
        "--visualization",
        choices=("rigid-body", "skeleton"),
        default="rigid-body",
        help="Motive 右侧面板的可视化类型",
    )
    parser.add_argument(
        "--rigid-body",
        help="Rigid Body 名称；仅有一个刚体时自动选择",
    )
    parser.add_argument(
        "--skeleton",
        help="Motive skeleton 命名空间；省略则选择 Bone 数最多的 skeleton",
    )
    parser.add_argument(
        "--include-unstable",
        action="store_true",
        help="也使用未通过 Time Echo 稳定性检查的记录",
    )
    parser.add_argument(
        "--allow-offset-extrapolation",
        action="store_true",
        help="允许在 Time Echo 范围外沿用最近端点 offset（不推荐）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 MP4；默认写入项目 sync/ 文件夹",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的输出 MP4 和 JSON 报告",
    )
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument(
        "--output-fps",
        type=float,
        help="省略时使用 Neon scene video FPS",
    )
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--max-duration",
        type=float,
        help="只渲染开头指定秒数；适合快速预览",
    )
    parser.add_argument(
        "--local-timezone",
        default="Europe/Berlin",
        help="视频中 Ubuntu local 时间的显示时区",
    )
    return parser


def default_output_path(motive_path: Path, recording_dir: Path) -> Path:
    safe_neon_name = recording_dir.name.replace(" ", "_")
    return DEFAULT_OUTPUT_DIR / f"{motive_path.stem}__{safe_neon_name}_global_sync.mp4"


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.motive_ubuntu_offset_ms):
        raise ValueError("--motive-ubuntu-offset-ms 必须是有限数字")
    if args.output_fps is not None and args.output_fps <= 0:
        raise ValueError("--output-fps 必须为正")
    if args.max_duration is not None and args.max_duration <= 0:
        raise ValueError("--max-duration 必须为正")
    try:
        local_timezone = ZoneInfo(args.local_timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"系统中不存在时区 {args.local_timezone!r}") from exc

    output_path = (
        args.output.resolve()
        if args.output
        else default_output_path(
            args.motive_csv, args.neon_recording_dir
        ).resolve()
    )
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("--output 必须以 .mp4 结尾")
    report_path = output_path.with_suffix(".json")
    existing_outputs = [path for path in (output_path, report_path) if path.exists()]
    if existing_outputs and not args.overwrite:
        raise ValueError(
            "输出已存在；请更换 --output 或明确添加 --overwrite："
            + ", ".join(str(path) for path in existing_outputs)
        )
    input_paths = {
        args.motive_csv.resolve(),
        (args.neon_recording_dir / "info.json").resolve(),
    }
    if args.offset_log is not None:
        input_paths.add(args.offset_log.resolve())
    for optional in (args.neon_video, args.neon_video_time):
        if optional is not None:
            input_paths.add(optional.resolve())
    if output_path in input_paths or report_path in input_paths:
        raise ValueError("输出路径不能覆盖输入文件")

    if args.offset_log is not None:
        clock_anchors = read_clock_anchors(
            args.offset_log, args.include_unstable
        )
        companion_to_ubuntu = make_clock_converter(
            clock_anchors,
            source_attr="companion_ns",
            target_attr="ubuntu_ns",
            allow_extrapolation=args.allow_offset_extrapolation,
        )
        neon_clock_mode = "time_echo_correction"
    else:
        if args.include_unstable or args.allow_offset_extrapolation:
            raise ValueError(
                "--include-unstable 和 --allow-offset-extrapolation 只有在提供 "
                "--offset-log 时才有意义"
            )
        clock_anchors = []
        companion_to_ubuntu = int
        neon_clock_mode = "assume_companion_equals_ubuntu"
        print(
            "提示：未提供 --offset-log；不校正 Neon Companion 与 Ubuntu "
            "的时钟误差。"
        )
    motive = load_motive_motion(
        args.motive_csv,
        args.motive_timezone,
        args.motive_ubuntu_offset_ms,
        args.visualization,
        args.rigid_body,
        args.skeleton,
    )
    neon = load_neon_video(
        args.neon_recording_dir,
        args.neon_video,
        args.neon_video_time,
        companion_to_ubuntu,
    )
    output_fps = args.output_fps or neon.fps
    render_result = render_video(
        motive,
        neon,
        output_path,
        args.output_width,
        args.output_height,
        output_fps,
        args.codec,
        args.max_duration,
        local_timezone,
    )

    report = {
        "schema_version": 1,
        "method": "global_start_time_only",
        "output_mp4": str(output_path),
        "inputs": {
            "motive_csv": str(motive.path.resolve()),
            "neon_recording_dir": str(neon.recording_dir.resolve()),
            "neon_video": str(neon.video_path.resolve()),
            "neon_video_time": (
                str(neon.time_path.resolve()) if neon.time_path else None
            ),
            "time_echo_log": (
                str(args.offset_log.resolve()) if args.offset_log else None
            ),
        },
        "motive": {
            "take_name": motive.take_name,
            "capture_start_text": motive.capture_start_text,
            "capture_start_interpreted_timezone": args.motive_timezone,
            "ubuntu_minus_windows_offset_ms": args.motive_ubuntu_offset_ms,
            "capture_start_ubuntu_ns": motive.capture_start_ubuntu_ns,
            "capture_start_utc": utc_iso(motive.capture_start_ubuntu_ns),
            "frame_rate_hz": motive.frame_rate_hz,
            "frame_count": len(motive.frame_ids),
            "visualization_mode": motive.visualization_mode,
            "visualized_object": motive.object_name,
            "skeleton_namespace": motive.skeleton_namespace,
            "source_units": motive.source_units,
        },
        "neon": {
            "recording_name": neon.recording_name,
            "recording_start_companion_ns": neon.recording_start_companion_ns,
            "recording_start_ubuntu_ns": neon.recording_start_ubuntu_ns,
            "recording_start_ubuntu_utc": utc_iso(
                neon.recording_start_ubuntu_ns
            ),
            "video_frame_count": len(neon.ubuntu_ns),
            "video_fps": neon.fps,
        },
        "time_echo": {
            "enabled": args.offset_log is not None,
            "clock_conversion_mode": neon_clock_mode,
            "used_anchor_count": len(clock_anchors),
            "include_unstable": bool(args.include_unstable),
            "allow_offset_extrapolation": bool(
                args.allow_offset_extrapolation
            ),
            "offset_definition": "Ubuntu - Neon Companion",
        },
        "render": render_result,
        "assumptions_and_limits": [
            "No motion or event anchor was used.",
            (
                "Time Echo corrects Neon Companion only; it does not measure "
                "the Motive Windows clock."
                if args.offset_log is not None
                else "No Time Echo correction was applied; Neon Companion "
                "timestamps were assumed equal to Ubuntu timestamps."
            ),
            (
                "Motive accuracy therefore depends on the declared timezone "
                "and Ubuntu-minus-Windows offset."
            ),
            "The MP4 has no audio.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"完成：{output_path}")
    print(f"同步报告：{report_path}")
    if args.offset_log is None:
        print("Neon 时钟：未校正（假定 Companion == Ubuntu）。")
    else:
        print(
            f"Neon 时钟：已使用 Time Echo 校正（{len(clock_anchors)} 个锚点）。"
        )
    print(
        "注意：本方法没有测量 Motive Windows→Ubuntu 的时钟误差；"
        f"当前使用 {args.motive_ubuntu_offset_ms:+.3f} ms。"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
