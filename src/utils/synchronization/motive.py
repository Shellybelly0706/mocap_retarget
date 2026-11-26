"""Reusable synchronization motive helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

from utils.synchronization.models import (
    MotiveExport,
)


def normalized(value: str) -> str:
    return value.strip().lower()


def padded_cell(row: Sequence[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def metadata_pairs(rows: Sequence[Sequence[str]], before: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows[:before]:
        for index in range(0, len(row) - 1, 2):
            key = row[index].strip()
            if key:
                result[normalized(key)] = row[index + 1].strip()
    return result


def read_motive_csv(path: Path) -> MotiveExport:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"Motive CSV 为空：{path}")

    axis_rows = [
        index
        for index, row in enumerate(rows)
        if normalized(padded_cell(row, 0)) == "frame"
        and normalized(padded_cell(row, 1)) in {"time", "time (seconds)"}
    ]
    if len(axis_rows) != 1:
        raise ValueError(
            "无法唯一确定 Motive 的 Frame/Time 数据表头；"
            f"找到 {len(axis_rows)} 个候选"
        )
    axis_row = axis_rows[0]
    metadata = metadata_pairs(rows, axis_row)

    rate_text = metadata.get("export frame rate") or metadata.get("capture frame rate")
    if rate_text is None:
        raise ValueError("Motive CSV 缺少 Export/Capture Frame Rate")
    frame_rate = float(rate_text)
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"无效的 Motive 帧率：{rate_text!r}")

    frame_ids: list[int] = []
    for row_number, row in enumerate(rows[axis_row + 1 :], axis_row + 2):
        value = padded_cell(row, 0)
        if not value:
            continue
        try:
            frame_ids.append(int(value))
        except ValueError as exc:
            raise ValueError(
                f"Motive CSV 第 {row_number} 行 Frame 不是整数：{value!r}"
            ) from exc
    if not frame_ids:
        raise ValueError("Motive CSV 中没有数据帧")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Motive CSV 含重复 Frame 值")
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("Motive CSV 的 Frame 必须严格递增")

    capture_start: int | None = None
    capture_start_text = metadata.get("capture start frame")
    if capture_start_text:
        try:
            capture_start = int(capture_start_text)
        except ValueError:
            pass

    return MotiveExport(
        path=path,
        frame_rate_hz=frame_rate,
        frame_ids=tuple(frame_ids),
        first_frame=frame_ids[0],
        capture_start_frame=capture_start,
        take_name=metadata.get("take name") or None,
    )


def _find_header_row(rows: Sequence[Sequence[str]], label: str, before: int) -> int:
    wanted = normalized(label)
    matches = [
        index
        for index, row in enumerate(rows[:before])
        if any(normalized(cell) == wanted for cell in row[:2])
    ]
    if not matches:
        raise ValueError(f"Motive CSV 缺少 {label!r} 表头行")
    return matches[-1]
