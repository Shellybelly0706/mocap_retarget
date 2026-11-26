"""Reusable synchronization motive motion helpers."""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from utils.optitrack.skeleton import SKELETON_PARENTS
from utils.synchronization.video_models import (
    MotiveMotion,
)
from utils.synchronization.video_settings import (
    NS_PER_SECOND,
)


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
        delta.days * 86400 + delta.seconds
    ) * NS_PER_SECOND + delta.microseconds * 1000


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
        if normalized(padded_cell(row, 0)) == "frame" and normalized(
            padded_cell(row, 1)
        ) in {"time", "time (seconds)"}:
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
            exact = [
                name for name in namespaces if normalized(name) == normalized(requested)
            ]
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
            row for row in header if any(normalized(cell) == "position" for cell in row)
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
            if object_type == "Rigid Body" and (object_type, name) in complete_rotations
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
            namespace = select_skeleton_namespace(complete_bones, requested_skeleton)
            selected = sorted(
                name for name in complete_bones if name.rpartition(":")[0] == namespace
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
