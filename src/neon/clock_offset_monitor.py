"""Continuously estimate the clock offset between Ubuntu and Neon Companion.

The recording itself remains on the Companion phone and can be controlled with
Neon Monitor. This monitor only records a time-varying mapping between the
phone clock and the Ubuntu clock so that exported Neon timestamps can later be
converted into the Ubuntu time domain.

Pupil Labs Time Echo defines the offset as::

    offset = Ubuntu (client) time - Companion (host) time

Consequently, convert an exported Neon timestamp with::

    ubuntu_timestamp_ns = neon_timestamp_ns + offset_ms * 1_000_000

Time Echo timestamps have millisecond resolution. Repeated measurements and
low-roundtrip filtering improve robustness, but they cannot guarantee absolute
sub-millisecond synchronization over Wi-Fi.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import signal
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "time_sync"
SCHEMA_VERSION = 1


def to_jsonable(value: Any) -> Any:
    """Convert API values and datetimes into values accepted by ``json``."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "_asdict"):
        return {key: to_jsonable(item) for key, item in value._asdict().items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Iterable):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write human-readable JSON metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class JsonlWriter:
    """Line-buffered writer so every completed estimate is immediately durable."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8", buffering=1)

    def write(self, value: Mapping[str, Any]) -> None:
        json.dump(
            to_jsonable(value),
            self._file,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._file.write("\n")

    def close(self) -> None:
        self._file.close()


def read_device_info(device: Any) -> dict[str, Any]:
    """Read useful identifiers without failing on API-version differences."""

    names = (
        "phone_name",
        "phone_ip",
        "phone_id",
        "module_serial",
        "serial_number_glasses",
        "version_glasses",
    )
    info: dict[str, Any] = {}
    for name in names:
        try:
            info[name] = getattr(device, name)
        except Exception:
            info[name] = None
    return info


def connect_device(ip: str | None, port: int, discovery_timeout: float) -> Any:
    """Connect by address or discover the first Companion on the LAN."""

    try:
        from pupil_labs.realtime_api.simple import Device, discover_one_device
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pupil-labs-realtime-api；请使用根目录 environment.yml "
            "更新并激活 mocap_data 环境。"
        ) from exc

    if ip:
        print(f"正在连接 Neon Companion {ip}:{port} ...")
        return Device(address=ip, port=port)

    print(f"正在局域网中搜索 Neon Companion（最多 {discovery_timeout:g} 秒）...")
    device = discover_one_device(max_search_duration_seconds=discovery_timeout)
    if device is None:
        raise RuntimeError(
            "没有发现设备。请确认 Companion App 正在运行、手机与 Ubuntu 在同一"
            "局域网；也可使用 --ip 指定手机 IP。"
        )
    return device


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize_estimate(
    estimate: Any,
    *,
    best_fraction: float,
    stability_threshold_ms: float,
    max_roundtrip_ms: float,
) -> dict[str, Any]:
    """Create a robust summary from one batch of Time Echo measurements.

    The recommended offset is the median offset among the measurements with the
    lowest round-trip durations. Selecting low-RTT samples reduces the effect
    of transient Wi-Fi queueing. Raw measurements are retained for auditing.
    """

    offsets = [float(value) for value in estimate.time_offset_ms.measurements]
    roundtrips = [
        float(value) for value in estimate.roundtrip_duration_ms.measurements
    ]
    if not offsets or len(offsets) != len(roundtrips):
        raise ValueError("Time Echo 返回了空的或长度不一致的测量结果")

    pairs = sorted(zip(roundtrips, offsets, strict=True), key=lambda pair: pair[0])
    selected_count = min(
        len(pairs), max(2, math.ceil(len(pairs) * best_fraction))
    )
    selected = pairs[:selected_count]
    selected_roundtrips = [pair[0] for pair in selected]
    selected_offsets = [pair[1] for pair in selected]

    recommended_offset_ms = statistics.median(selected_offsets)
    selected_offset_std_ms = _sample_std(selected_offsets)
    median_roundtrip_ms = statistics.median(selected_roundtrips)
    stability_checks = {
        "selected_offset_std_within_threshold": (
            selected_offset_std_ms <= stability_threshold_ms
        ),
        "selected_median_roundtrip_within_threshold": (
            median_roundtrip_ms <= max_roundtrip_ms
        ),
    }

    return {
        "measurement_count": len(offsets),
        "recommended_ubuntu_minus_companion_ms": recommended_offset_ms,
        "conversion": {
            "neon_to_ubuntu": "ubuntu_ns = neon_ns + offset_ms * 1000000",
            "ubuntu_to_neon": "neon_ns = ubuntu_ns - offset_ms * 1000000",
        },
        "selection": {
            "method": "median_offset_of_lowest_roundtrip_measurements",
            "best_fraction": best_fraction,
            "selected_count": selected_count,
        },
        "selected_measurements": {
            "offset_mean_ms": statistics.fmean(selected_offsets),
            "offset_median_ms": recommended_offset_ms,
            "offset_std_ms": selected_offset_std_ms,
            "offset_min_ms": min(selected_offsets),
            "offset_max_ms": max(selected_offsets),
            "roundtrip_mean_ms": statistics.fmean(selected_roundtrips),
            "roundtrip_median_ms": median_roundtrip_ms,
            "roundtrip_min_ms": min(selected_roundtrips),
            "roundtrip_max_ms": max(selected_roundtrips),
        },
        "all_measurements": {
            "offset_mean_ms": statistics.fmean(offsets),
            "offset_median_ms": statistics.median(offsets),
            "offset_std_ms": _sample_std(offsets),
            "roundtrip_mean_ms": statistics.fmean(roundtrips),
            "roundtrip_median_ms": statistics.median(roundtrips),
            "roundtrip_min_ms": min(roundtrips),
            "roundtrip_max_ms": max(roundtrips),
        },
        "quality": {
            "stable_for_interpolation": all(stability_checks.values()),
            "checks": stability_checks,
            "stability_threshold_ms": stability_threshold_ms,
            "max_roundtrip_ms": max_roundtrip_ms,
            "warning": (
                "Quality checks describe repeatability and network latency; they "
                "do not prove absolute sub-millisecond clock accuracy."
            ),
        },
        "raw_measurements": {
            "ubuntu_minus_companion_ms": offsets,
            "roundtrip_ms": roundtrips,
        },
    }


def make_session_dir(output: Path | None, started_at: dt.datetime) -> Path:
    if output is not None:
        return output
    stamp = started_at.astimezone().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / stamp


def run(args: argparse.Namespace) -> int:
    """Run continuous clock-offset estimation until stopped."""

    started_at = dt.datetime.now(dt.timezone.utc)
    output_dir = make_session_dir(args.output, started_at)
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets_path = output_dir / "clock_offsets.jsonl"
    session_path = output_dir / "session.json"

    stop_requested = False

    def request_stop(*_: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    device: Any = None
    writer: JsonlWriter | None = None
    successful_offsets: list[float] = []
    success_count = 0
    failure_count = 0
    consecutive_failures = 0
    exit_code = 0
    device_info: dict[str, Any] = {}

    session: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Neon Companion to Ubuntu clock-offset monitoring",
        "started_at_utc": started_at,
        "output_files": {"clock_offsets": offsets_path.name},
        "clock_domains": {
            "client": "Ubuntu Unix/UTC system clock",
            "host": "Neon Companion Unix/UTC system clock",
            "offset_definition": "Ubuntu time - Neon Companion time",
        },
        "conversion": {
            "neon_to_ubuntu": "ubuntu_ns = neon_ns + offset_ms * 1000000",
            "ubuntu_to_neon": "neon_ns = ubuntu_ns - offset_ms * 1000000",
        },
        "configuration": {
            "requested_ip": args.ip,
            "port": args.port,
            "discovery_timeout_seconds": args.discovery_timeout,
            "interval_seconds": args.interval,
            "measurements_per_estimate": args.measurements,
            "sleep_between_measurements_seconds": args.measurement_sleep,
            "best_fraction": args.best_fraction,
            "stability_threshold_ms": args.stability_threshold_ms,
            "max_roundtrip_ms": args.max_roundtrip_ms,
            "duration_seconds": args.duration,
        },
        "important_note": (
            "This program estimates a mapping between clocks. It does not change "
            "either device's system clock and cannot guarantee <1 ms absolute error."
        ),
    }
    write_json(session_path, session)

    try:
        device = connect_device(args.ip, args.port, args.discovery_timeout)
        device_info = read_device_info(device)
        session["device"] = device_info
        write_json(session_path, session)

        identifier = device_info.get("module_serial") or device_info.get(
            "serial_number_glasses"
        )
        print(
            f"已连接：{device_info.get('phone_name') or 'Neon Companion'} "
            f"({device_info.get('phone_ip') or args.ip or 'IP unknown'}), "
            f"模块序列号={identifier or 'unknown'}"
        )
        print(f"偏移记录：{offsets_path.resolve()}")
        print(
            "约定：offset = Ubuntu - Companion；将导出的 Neon 时间戳转换到 "
            "Ubuntu 时间域时应加上 offset。"
        )
        print(
            "提示：Time Echo 为毫秒分辨率；稳定性检查不能保证绝对误差小于 1 ms。"
        )
        print("持续测量中；请在 Neon Monitor 中独立开始/停止录制，按 Ctrl-C 结束。")

        writer = JsonlWriter(offsets_path)
        deadline = time.monotonic() + args.duration if args.duration else None
        next_cycle = time.monotonic()
        sequence = 0

        while not stop_requested:
            if deadline is not None and time.monotonic() >= deadline:
                break

            wait_seconds = next_cycle - time.monotonic()
            if wait_seconds > 0:
                time.sleep(min(wait_seconds, 0.2))
                continue

            sequence += 1
            batch_started_ns = time.time_ns()
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "sequence": sequence,
                "batch_started_ubuntu_unix_ns": batch_started_ns,
                "batch_started_utc": dt.datetime.fromtimestamp(
                    batch_started_ns / 1e9, tz=dt.timezone.utc
                ),
            }

            try:
                estimate = device.estimate_time_offset(
                    number_of_measurements=args.measurements,
                    sleep_between_measurements_seconds=args.measurement_sleep,
                )
                if estimate is None:
                    raise RuntimeError(
                        "Time Echo 测量失败；请检查 Companion App 版本和网络连接"
                    )

                summary = summarize_estimate(
                    estimate,
                    best_fraction=args.best_fraction,
                    stability_threshold_ms=args.stability_threshold_ms,
                    max_roundtrip_ms=args.max_roundtrip_ms,
                )
                batch_finished_ns = time.time_ns()
                batch_midpoint_ns = (batch_started_ns + batch_finished_ns) // 2
                offset_ms = summary["recommended_ubuntu_minus_companion_ms"]
                companion_anchor_ns = batch_midpoint_ns - round(offset_ms * 1e6)
                record.update(
                    {
                        "status": "ok",
                        "batch_finished_ubuntu_unix_ns": batch_finished_ns,
                        "batch_midpoint_ubuntu_unix_ns": batch_midpoint_ns,
                        "estimated_companion_unix_ns_at_batch_midpoint": (
                            companion_anchor_ns
                        ),
                        "interpolation_anchor": {
                            "ubuntu_unix_ns": batch_midpoint_ns,
                            "companion_unix_ns": companion_anchor_ns,
                            "ubuntu_minus_companion_ms": offset_ms,
                        },
                        "batch_duration_ms": (
                            batch_finished_ns - batch_started_ns
                        )
                        / 1e6,
                        **summary,
                    }
                )
                writer.write(record)

                successful_offsets.append(float(offset_ms))
                success_count += 1
                consecutive_failures = 0
                quality = summary["quality"]["stable_for_interpolation"]
                selected = summary["selected_measurements"]
                print(
                    f"[{sequence:06d}] offset={offset_ms:+.3f} ms "
                    f"selected_std={selected['offset_std_ms']:.3f} ms "
                    f"RTT_median={selected['roundtrip_median_ms']:.3f} ms "
                    f"stable={'yes' if quality else 'NO'}"
                )
            except Exception as exc:
                failure_count += 1
                consecutive_failures += 1
                record.update(
                    {
                        "status": "error",
                        "batch_finished_ubuntu_unix_ns": time.time_ns(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                writer.write(record)
                print(f"[{sequence:06d}] 测量失败：{exc}", file=sys.stderr)

                if consecutive_failures >= args.reconnect_after_failures:
                    print("连续测量失败，重新连接 Companion ...", file=sys.stderr)
                    try:
                        device.close()
                    except Exception:
                        pass
                    device = connect_device(
                        args.ip, args.port, args.discovery_timeout
                    )
                    device_info = read_device_info(device)
                    consecutive_failures = 0

            next_cycle = max(next_cycle + args.interval, time.monotonic())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        session["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 1
    finally:
        if writer is not None:
            writer.close()
        if device is not None:
            try:
                device.close()
            except Exception as exc:
                print(f"警告：关闭设备连接时出错：{exc}", file=sys.stderr)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

        session["stopped_at_utc"] = dt.datetime.now(dt.timezone.utc)
        session["result"] = {
            "successful_estimates": success_count,
            "failed_estimates": failure_count,
        }
        if successful_offsets:
            session["result"].update(
                {
                    "first_recommended_offset_ms": successful_offsets[0],
                    "last_recommended_offset_ms": successful_offsets[-1],
                    "minimum_recommended_offset_ms": min(successful_offsets),
                    "maximum_recommended_offset_ms": max(successful_offsets),
                    "offset_change_first_to_last_ms": (
                        successful_offsets[-1] - successful_offsets[0]
                    ),
                }
            )
        write_json(session_path, session)

    print(
        f"已停止。成功估计 {success_count} 次，失败 {failure_count} 次；"
        f"日志目录：{output_dir.resolve()}"
    )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "持续估计并记录 Ubuntu 与 Pupil Labs Neon Companion 的时钟偏移"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", help="Companion 手机 IP；省略则通过 mDNS 自动发现")
    parser.add_argument("--port", type=int, default=8080, help="Companion API 端口")
    parser.add_argument(
        "--discovery-timeout", type=float, default=10.0, help="设备发现超时（秒）"
    )
    parser.add_argument(
        "--interval", type=float, default=5.0, help="两批偏移估计开始时间的间隔（秒）"
    )
    parser.add_argument(
        "--measurements", type=int, default=100, help="每批 Time Echo 请求数量"
    )
    parser.add_argument(
        "--measurement-sleep",
        type=float,
        default=None,
        help="同一批中每次 Time Echo 请求之间的等待时间（秒）",
    )
    parser.add_argument(
        "--best-fraction",
        type=float,
        default=0.25,
        help="按 RTT 排序后用于推荐偏移的最低延迟样本比例",
    )
    parser.add_argument(
        "--stability-threshold-ms",
        type=float,
        default=1.0,
        help="低 RTT 样本偏移标准差的稳定性阈值（不是绝对精度保证）",
    )
    parser.add_argument(
        "--max-roundtrip-ms",
        type=float,
        default=5.0,
        help="低 RTT 样本中位往返延迟的质量阈值",
    )
    parser.add_argument(
        "--reconnect-after-failures",
        type=int,
        default=3,
        help="连续失败多少批后重新连接设备",
    )
    parser.add_argument("--duration", type=float, help="运行时间（秒）；省略则持续运行")
    parser.add_argument(
        "--output", type=Path, help="本次同步日志目录；省略则按开始时间自动创建"
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    if args.discovery_timeout <= 0:
        parser.error("--discovery-timeout 必须大于 0")
    if args.interval <= 0:
        parser.error("--interval 必须大于 0")
    if args.measurements < 2:
        parser.error("--measurements 必须至少为 2")
    if args.measurement_sleep is not None and args.measurement_sleep < 0:
        parser.error("--measurement-sleep 不能小于 0")
    if not 0 < args.best_fraction <= 1:
        parser.error("--best-fraction 必须在 (0, 1] 范围内")
    if args.stability_threshold_ms < 0:
        parser.error("--stability-threshold-ms 不能小于 0")
    if args.max_roundtrip_ms < 0:
        parser.error("--max-roundtrip-ms 不能小于 0")
    if args.reconnect_after_failures < 1:
        parser.error("--reconnect-after-failures 必须至少为 1")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须大于 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
