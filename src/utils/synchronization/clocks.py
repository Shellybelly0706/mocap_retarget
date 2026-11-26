"""Reusable synchronization clocks helpers."""

from __future__ import annotations

import bisect
import datetime as dt
import json
import re
import struct
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Sequence

from utils.synchronization.constants import (
    NS_PER_SECOND,
)
from utils.synchronization.models import (
    ClockAnchor,
)

ISO_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<timezone>Z|[+-]\d{2}:\d{2})$"
)


def read_clock_anchors(path: Path, include_unstable: bool) -> list[ClockAnchor]:
    anchors: list[ClockAnchor] = []
    success_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"offset 日志第 {line_number} 行不是合法 JSON"
                ) from exc
            if record.get("status") != "ok":
                continue
            success_count += 1
            raw = record.get("interpolation_anchor")
            if not isinstance(raw, dict):
                continue
            quality = record.get("quality")
            stable = bool(
                isinstance(quality, dict)
                and quality.get("stable_for_interpolation", False)
            )
            if not include_unstable and not stable:
                continue
            try:
                companion_ns = int(raw["companion_unix_ns"])
                ubuntu_ns = int(raw["ubuntu_unix_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"offset 日志第 {line_number} 行的 interpolation_anchor 无效"
                ) from exc
            anchors.append(
                ClockAnchor(
                    companion_ns=companion_ns,
                    ubuntu_ns=ubuntu_ns,
                    stable=stable,
                    sequence=record.get("sequence"),
                )
            )

    if not anchors:
        detail = (
            "没有通过 stable_for_interpolation 检查的记录；若确认仍要使用，"
            "可添加 --include-unstable"
            if success_count and not include_unstable
            else "没有可用的成功记录"
        )
        raise ValueError(f"offset 日志中{detail}：{path}")

    anchors.sort(key=lambda item: item.companion_ns)
    deduplicated: list[ClockAnchor] = []
    for anchor in anchors:
        if deduplicated and anchor.companion_ns == deduplicated[-1].companion_ns:
            if anchor.ubuntu_ns != deduplicated[-1].ubuntu_ns:
                raise ValueError(
                    "offset 日志含同一 Companion 时间但 Ubuntu 时间不同的锚点"
                )
            continue
        if deduplicated and anchor.ubuntu_ns <= deduplicated[-1].ubuntu_ns:
            raise ValueError("offset 日志锚点不是单调时钟映射")
        deduplicated.append(anchor)
    return deduplicated


def _rounded_ratio(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest, with symmetric handling of signs."""

    if denominator <= 0:
        raise ValueError("插值分母必须为正")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def make_clock_converter(
    anchors: Sequence[ClockAnchor],
    *,
    source_attr: str,
    target_attr: str,
    allow_extrapolation: bool,
) -> Callable[[int], int]:
    source = [int(getattr(anchor, source_attr)) for anchor in anchors]
    target = [int(getattr(anchor, target_attr)) for anchor in anchors]

    def convert(timestamp_ns: int) -> int:
        position = bisect.bisect_left(source, timestamp_ns)
        if position < len(source) and source[position] == timestamp_ns:
            return target[position]
        if position == 0:
            if not allow_extrapolation:
                raise ValueError(
                    f"时间戳 {timestamp_ns} 早于 offset 日志覆盖范围 "
                    f"[{source[0]}, {source[-1]}]"
                )
            return timestamp_ns + (target[0] - source[0])
        if position == len(source):
            if not allow_extrapolation:
                raise ValueError(
                    f"时间戳 {timestamp_ns} 晚于 offset 日志覆盖范围 "
                    f"[{source[0]}, {source[-1]}]"
                )
            return timestamp_ns + (target[-1] - source[-1])

        source_left, source_right = source[position - 1], source[position]
        target_left, target_right = target[position - 1], target[position]
        delta = _rounded_ratio(
            (timestamp_ns - source_left) * (target_right - target_left),
            source_right - source_left,
        )
        return target_left + delta

    return convert


def parse_timestamp_ns(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("Neon 时间戳为空")
    try:
        if any(character in text for character in ".eE"):
            seconds = Decimal(text)
            if not seconds.is_finite() or seconds < 100_000_000:
                raise ValueError
            return int(
                (seconds * NS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_UP)
            )
        integer = int(text)
        if integer < 100_000_000_000_000_000:
            raise ValueError(
                "整数 Neon 时间戳应为 Unix 纳秒（通常为 19 位）；"
                "Unix 秒请写成带小数形式"
            )
        return integer
    except (ValueError, InvalidOperation):
        match = ISO_TIMESTAMP_RE.fullmatch(text)
        if match is not None:
            timezone = (
                "+00:00" if match.group("timezone") == "Z" else match.group("timezone")
            )
            whole_seconds = dt.datetime.fromisoformat(
                f"{match.group('date')}T{match.group('time')}{timezone}"
            )
            seconds_ns = int(whole_seconds.timestamp()) * NS_PER_SECOND
            fraction_ns = int((match.group("fraction") or "0").ljust(9, "0"))
            return seconds_ns + fraction_ns
        else:
            raise ValueError(
                f"无法解析 Neon 时间戳 {value!r}；请使用 Unix 纳秒、"
                "带小数的 Unix 秒或带时区 ISO-8601"
            ) from None


def parse_pair(value: str, right_parser: Callable[[str], int]) -> tuple[int, int]:
    if "=" not in value:
        raise ValueError(f"锚点 {value!r} 应使用 MOTIVE_FRAME=NEON_VALUE 格式")
    left, right = value.split("=", 1)
    try:
        motive_frame = int(left.strip())
    except ValueError as exc:
        raise ValueError(f"Motive frame 不是整数：{left!r}") from exc
    return motive_frame, right_parser(right.strip())


def read_neon_time_file(path: Path) -> tuple[int, ...]:
    size = path.stat().st_size
    if size % 8:
        raise ValueError(f"Neon .time 文件大小不是 8 字节的整数倍：{path}")
    timestamps: list[int] = []
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 65536):
            count = len(chunk) // 8
            timestamps.extend(struct.unpack(f"<{count}q", chunk))
    if not timestamps:
        raise ValueError(f"Neon .time 文件为空：{path}")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Neon .time 时间戳不是单调递增：{path}")
    return tuple(timestamps)


def utc_iso(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, NS_PER_SECOND)
    base = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    return f"{base:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"
