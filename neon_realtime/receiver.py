"""Receive Pupil Labs Neon data over Wi-Fi and save it locally.

The Pupil Labs dependency is imported lazily so that ``--help`` and the unit
tests also work before the hardware-specific dependencies have been installed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import json
import math
import queue
import signal
import statistics
import struct
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, TextIO


STREAM_CHOICES = ("gaze", "imu", "events", "scene")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "neon_data"

IMU_DTYPE_TEXT = """[
("timestamp_ns", "int64"),
("gyro_x", "float32"),
("gyro_y", "float32"),
("gyro_z", "float32"),
("accel_x", "float32"),
("accel_y", "float32"),
("accel_z", "float32"),
("quaternion_w", "float32"),
("quaternion_x", "float32"),
("quaternion_y", "float32"),
("quaternion_z", "float32"),
]
"""

IMU_PROTO_TEXT = """syntax = "proto3";
package imuproto;
message AccelData {
     float x = 1;
     float y = 2;
     float z = 3;
     int32 reserved = 4;
}
message GyroData {
     float x = 1;
     float y = 2;
     float z = 3;
     int32 reserved = 4;
}
message QuaternionData {
     float w = 1;
     float x = 2;
     float y = 3;
     float z = 4;
     float reserved = 5;
}
message ImuPacket {
     uint64 tsNs = 1;
     AccelData accelData = 2;
     GyroData gyroData = 3;
     QuaternionData quaternionData = 4;
}
"""


class DeviceLike(Protocol):
    """Subset of the simple API used by the receiver."""

    def receive_gaze_datum(self, timeout_seconds: float | None = None) -> Any: ...

    def receive_imu_datum(self, timeout_seconds: float | None = None) -> Any: ...

    def receive_eye_events(self, timeout_seconds: float | None = None) -> Any: ...

    def receive_matched_scene_video_frame_and_gaze(
        self, timeout_seconds: float | None = None
    ) -> Any: ...

    def close(self) -> None: ...


def to_jsonable(value: Any) -> Any:
    """Recursively convert Pupil Labs named tuples and NumPy values to JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "_asdict"):
        return {key: to_jsonable(item) for key, item in value._asdict().items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
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


class JsonlWriter:
    """Small line-buffered JSONL writer."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("w", encoding="utf-8", buffering=1)

    def write(self, stream: str, datum: Any) -> None:
        record = {
            "stream": stream,
            "received_at_unix_ns": time.time_ns(),
            "data_type": type(datum).__name__,
            "data": to_jsonable(datum),
        }
        json.dump(record, self._file, ensure_ascii=False, separators=(",", ":"))
        self._file.write("\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ExportCompatibleImuWriter:
    """Write live IMU packets in both JSONL and Companion Export layouts."""

    _decoded_struct = struct.Struct("<q10f")
    _timestamp_struct = struct.Struct("<q")
    _payload_length_struct = struct.Struct("<H")

    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        export_dir = output_dir / "imu"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "imu.dtype").write_text(IMU_DTYPE_TEXT, encoding="utf-8")
        (export_dir / "imu.proto").write_text(IMU_PROTO_TEXT, encoding="utf-8")
        self._jsonl = (output_dir / "imu.jsonl").open(
            "w", encoding="utf-8", buffering=1024 * 1024
        )
        self._decoded = (export_dir / "imu ps1.raw").open("wb", buffering=1024 * 1024)
        self._decoded_time = (export_dir / "imu ps1.time").open(
            "wb", buffering=1024 * 1024
        )
        self._encoded = (export_dir / "extimu ps1.raw").open(
            "wb", buffering=1024 * 1024
        )
        self._encoded_time = (export_dir / "extimu ps1.time").open(
            "wb", buffering=1024 * 1024
        )

    def write(
        self,
        raw_payload: bytes,
        timestamp_ns: int,
        gyro: tuple[float, float, float],
        accel: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
    ) -> None:
        received_at_ns = time.time_ns()
        record = {
            "stream": "imu",
            "received_at_unix_ns": received_at_ns,
            "data_type": "IMUData",
            "data": {
                "gyro_data": dict(zip(("x", "y", "z"), gyro, strict=True)),
                "accel_data": dict(zip(("x", "y", "z"), accel, strict=True)),
                "quaternion": {
                    "x": quaternion_wxyz[1],
                    "y": quaternion_wxyz[2],
                    "z": quaternion_wxyz[3],
                    "w": quaternion_wxyz[0],
                },
                "timestamp_unix_seconds": timestamp_ns / 1e9,
                "timestamp_unix_ns": timestamp_ns,
            },
        }
        self._jsonl.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )

        timestamp_bytes = self._timestamp_struct.pack(timestamp_ns)
        self._decoded.write(
            self._decoded_struct.pack(
                timestamp_ns, *gyro, *accel, *quaternion_wxyz
            )
        )
        self._decoded_time.write(timestamp_bytes)
        self._encoded.write(self._payload_length_struct.pack(len(raw_payload)))
        self._encoded.write(raw_payload)
        self._encoded_time.write(timestamp_bytes)

    def close(self) -> None:
        for file in (
            self._jsonl,
            self._decoded,
            self._decoded_time,
            self._encoded,
            self._encoded_time,
        ):
            file.close()

    def __enter__(self) -> ExportCompatibleImuWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class Counters:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, name: str) -> int:
        with self._lock:
            self._values[name] = self._values.get(name, 0) + 1
            return self._values[name]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)


def _compact_preview(stream: str, datum: Any) -> str:
    data = to_jsonable(datum)
    if stream == "gaze" and isinstance(data, dict):
        return (
            f"x={data.get('x', '?')} y={data.get('y', '?')} "
            f"worn={data.get('worn', '?')}"
        )
    if stream == "imu" and isinstance(data, dict):
        return f"gyro={data.get('gyro_data', '?')} accel={data.get('accel_data', '?')}"
    if stream == "events" and isinstance(data, dict):
        return f"type={data.get('event_type', type(datum).__name__)}"
    return type(datum).__name__


def receive_data_stream(
    device: DeviceLike,
    stream: str,
    method_name: str,
    stop_event: threading.Event,
    output_dir: Path | None,
    counters: Counters,
    print_interval: float,
) -> None:
    method = getattr(device, method_name)
    writer = JsonlWriter(output_dir / f"{stream}.jsonl") if output_dir else None
    next_print = 0.0
    try:
        while not stop_event.is_set():
            datum = method(timeout_seconds=0.5)
            if datum is None:
                continue
            count = counters.increment(stream)
            if writer:
                writer.write(stream, datum)
            now = time.monotonic()
            if print_interval > 0 and now >= next_print:
                print(f"[{stream:6}] #{count:<7} {_compact_preview(stream, datum)}")
                next_print = now + print_interval
    finally:
        if writer:
            writer.close()


def _write_imu_quality_report(output_dir: Path, timestamps_ns: list[int]) -> None:
    report: dict[str, Any] = {
        "receiver": "direct_async_rtsp",
        "sample_count": len(timestamps_ns),
        "note": (
            "Wi-Fi/RTSP cannot guarantee zero packet loss. estimated_missing_samples "
            "is inferred from timestamp gaps."
        ),
    }
    if len(timestamps_ns) >= 2:
        intervals = [
            current - previous
            for previous, current in zip(timestamps_ns, timestamps_ns[1:])
        ]
        positive_intervals = [value for value in intervals if value > 0]
        duration_ns = timestamps_ns[-1] - timestamps_ns[0]
        report.update(
            {
                "first_timestamp_ns": timestamps_ns[0],
                "last_timestamp_ns": timestamps_ns[-1],
                "duration_seconds": duration_ns / 1e9,
                "effective_rate_hz": (
                    (len(timestamps_ns) - 1) * 1e9 / duration_ns
                    if duration_ns > 0
                    else None
                ),
                "maximum_interval_ms": max(intervals) / 1e6,
                "nonpositive_intervals": sum(value <= 0 for value in intervals),
            }
        )
        if positive_intervals:
            median_ns = statistics.median(positive_intervals)
            report.update(
                {
                    "median_interval_ms": median_ns / 1e6,
                    "estimated_missing_samples": sum(
                        max(0, round(value / median_ns) - 1)
                        for value in positive_intervals
                        if value > median_ns * 1.5
                    ),
                }
            )
    _write_session_metadata(output_dir / "imu" / "quality.json", report)


async def _receive_direct_imu_async(
    device: Any,
    stop_event: threading.Event,
    output_dir: Path | None,
    counters: Counters,
    print_interval: float,
) -> None:
    try:
        from pupil_labs.neon_recording.timeseries.imu.imu_pb2 import ImuPacket
        from pupil_labs.realtime_api.device import Device as AsyncDevice
        from pupil_labs.realtime_api.streaming import RTSPRawStreamer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Pupil Labs 实时 API；请运行 "
            "pip install -r neon_realtime/requirements.txt"
        ) from exc

    async with AsyncDevice(address=device.address, port=device.port) as control:
        status = await control.get_status()
        sensor = status.direct_imu_sensor()
        stream_url = sensor.url if sensor else None
    if not stream_url:
        raise RuntimeError("Neon Companion 没有提供已连接的 IMU RTSP 数据流")

    print(f"[imu   ] 使用无单元素缓存的异步 RTSP：{stream_url}")
    writer = ExportCompatibleImuWriter(output_dir) if output_dir else None
    timestamps_ns: list[int] = []
    next_print = 0.0
    next_item_task: asyncio.Task[Any] | None = None
    try:
        async with RTSPRawStreamer(stream_url) as streamer:
            iterator = streamer.receive().__aiter__()
            next_item_task = asyncio.create_task(anext(iterator))
            while not stop_event.is_set():
                done, _ = await asyncio.wait({next_item_task}, timeout=0.2)
                if not done:
                    continue
                try:
                    raw_item = next_item_task.result()
                except StopAsyncIteration:
                    break

                payload = bytes(raw_item.raw)
                packet = ImuPacket()
                packet.ParseFromString(payload)
                timestamp_ns = int(packet.tsNs)
                gyro = (
                    float(packet.gyroData.x),
                    float(packet.gyroData.y),
                    float(packet.gyroData.z),
                )
                accel = (
                    float(packet.accelData.x),
                    float(packet.accelData.y),
                    float(packet.accelData.z),
                )
                quaternion_wxyz = (
                    float(packet.rotVecData.w),
                    float(packet.rotVecData.x),
                    float(packet.rotVecData.y),
                    float(packet.rotVecData.z),
                )
                if output_dir:
                    timestamps_ns.append(timestamp_ns)
                count = counters.increment("imu")
                if writer:
                    writer.write(
                        payload, timestamp_ns, gyro, accel, quaternion_wxyz
                    )

                now = time.monotonic()
                if print_interval > 0 and now >= next_print:
                    print(
                        f"[imu   ] #{count:<7} gyro={gyro} accel={accel} "
                        f"timestamp_ns={timestamp_ns}"
                    )
                    next_print = now + print_interval
                next_item_task = asyncio.create_task(anext(iterator))
    finally:
        if next_item_task is not None and not next_item_task.done():
            next_item_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await next_item_task
        if writer:
            writer.close()
        if output_dir:
            _write_imu_quality_report(output_dir, timestamps_ns)


def receive_direct_imu_stream(
    device: Any,
    stop_event: threading.Event,
    output_dir: Path | None,
    counters: Counters,
    print_interval: float,
) -> None:
    asyncio.run(
        _receive_direct_imu_async(
            device, stop_event, output_dir, counters, print_interval
        )
    )


def receive_scene_stream(
    device: DeviceLike,
    stop_event: threading.Event,
    output_dir: Path | None,
    counters: Counters,
    preview: bool,
    fps: float,
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "场景视频需要 OpenCV；请运行 pip install -r neon_realtime/requirements.txt"
        ) from exc

    metadata = JsonlWriter(output_dir / "scene_gaze.jsonl") if output_dir else None
    video_writer: Any = None
    try:
        while not stop_event.is_set():
            matched = device.receive_matched_scene_video_frame_and_gaze(
                timeout_seconds=0.5
            )
            if matched is None:
                continue

            frame = matched.frame
            gaze = matched.gaze
            pixels = frame.bgr_pixels.copy()
            gaze_x = int(round(gaze.x))
            gaze_y = int(round(gaze.y))
            if getattr(gaze, "worn", True):
                cv2.circle(pixels, (gaze_x, gaze_y), 18, (0, 0, 255), 3)

            if output_dir and video_writer is None:
                height, width = pixels.shape[:2]
                video_path = output_dir / "scene.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, fps, (width, height)
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"无法创建视频文件：{video_path}")

            if video_writer is not None:
                video_writer.write(pixels)
            if metadata:
                metadata.write(
                    "scene",
                    {
                        "frame_timestamp_unix_seconds": frame.timestamp_unix_seconds,
                        "gaze": gaze,
                    },
                )
            count = counters.increment("scene")
            if count == 1 or count % max(1, round(fps * 5)) == 0:
                print(f"[scene ] #{count:<7} frame={pixels.shape[1]}x{pixels.shape[0]}")

            if preview:
                cv2.imshow("Pupil Labs Neon - scene + gaze", pixels)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    stop_event.set()
    finally:
        if video_writer is not None:
            video_writer.release()
        if metadata:
            metadata.close()
        if preview:
            cv2.destroyAllWindows()


def _read_device_info(device: Any) -> dict[str, Any]:
    names = (
        "phone_name",
        "phone_ip",
        "phone_id",
        "battery_level_percent",
        "battery_state",
        "memory_num_free_bytes",
        "memory_state",
        "module_serial",
        "serial_number_glasses",
        "serial_number_scene_cam",
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
    try:
        from pupil_labs.realtime_api.simple import Device, discover_one_device
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pupil-labs-realtime-api；请先运行 "
            "pip install -r neon_realtime/requirements.txt"
        ) from exc

    if ip:
        print(f"正在连接 Neon Companion {ip}:{port} ...")
        return Device(address=ip, port=port)

    print(f"正在局域网中搜索 Neon Companion（最多 {discovery_timeout:g} 秒）...")
    device = discover_one_device(max_search_duration_seconds=discovery_timeout)
    if device is None:
        raise RuntimeError(
            "没有发现设备。请确认 Companion App 正在运行且电脑和手机在同一局域网；"
            "也可使用 --ip 指定手机 IP。"
        )
    return device


def _write_session_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    device: Any = None
    stop_event = threading.Event()
    counters = Counters()
    errors: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    threads: list[threading.Thread] = []
    output_dir: Path | None = None
    started_at = dt.datetime.now(dt.timezone.utc)

    if not args.no_save:
        if args.output:
            output_dir = args.output
        else:
            stamp = started_at.astimezone().strftime("%Y%m%d_%H%M%S")
            output_dir = DEFAULT_OUTPUT_ROOT / stamp
        output_dir.mkdir(parents=True, exist_ok=True)

    def request_stop(*_: object) -> None:
        stop_event.set()

    def guarded_worker(name: str, target: Any, *worker_args: Any) -> None:
        try:
            target(*worker_args)
        except BaseException as exc:
            errors.put((name, exc))
            stop_event.set()

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    try:
        device = connect_device(args.ip, args.port, args.discovery_timeout)
        device_info = _read_device_info(device)
        identifier = device_info.get("module_serial") or device_info.get(
            "serial_number_glasses"
        )
        print(
            f"已连接：{device_info.get('phone_name') or 'Neon Companion'} "
            f"({device_info.get('phone_ip') or args.ip or 'IP unknown'}), "
            f"眼镜/模块序列号={identifier or 'unknown'}"
        )
        print("接收中；按 Ctrl-C 停止。")

        if output_dir:
            _write_session_metadata(
                output_dir / "session.json",
                {
                    "started_at_utc": started_at,
                    "streams": args.streams,
                    "imu_mode": args.imu_mode,
                    "device": device_info,
                    "connection": {"requested_ip": args.ip, "port": args.port},
                },
            )
            print(f"数据目录：{output_dir.resolve()}")

        methods = {
            "gaze": "receive_gaze_datum",
            "imu": "receive_imu_datum",
            "events": "receive_eye_events",
        }
        for stream in args.streams:
            if stream == "scene":
                worker_args = (
                    device,
                    stop_event,
                    output_dir,
                    counters,
                    args.preview,
                    args.video_fps,
                )
                target = receive_scene_stream
            elif stream == "imu" and args.imu_mode == "direct":
                worker_args = (
                    device,
                    stop_event,
                    output_dir,
                    counters,
                    args.print_interval,
                )
                target = receive_direct_imu_stream
            else:
                worker_args = (
                    device,
                    stream,
                    methods[stream],
                    stop_event,
                    output_dir,
                    counters,
                    args.print_interval,
                )
                target = receive_data_stream
            thread = threading.Thread(
                target=guarded_worker,
                args=(stream, target, *worker_args),
                name=f"neon-{stream}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        deadline = time.monotonic() + args.duration if args.duration else None
        while not stop_event.wait(0.2):
            if deadline is not None and time.monotonic() >= deadline:
                stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)
        if device is not None:
            device.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if not errors.empty():
        while not errors.empty():
            name, exc = errors.get_nowait()
            print(f"{name} 接收线程错误：{exc}", file=sys.stderr)
        return 1

    counts = counters.snapshot()
    if output_dir:
        session_path = output_dir / "session.json"
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
            session["stopped_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            session["received_counts"] = counts
            _write_session_metadata(session_path, session)
        except OSError as exc:
            print(f"警告：无法更新会话元数据：{exc}", file=sys.stderr)
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"已停止。接收计数：{summary or '无数据'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通过 Wi-Fi 实时接收 Pupil Labs Neon 数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", help="Neon Companion 手机 IP；省略则通过 mDNS 自动发现")
    parser.add_argument("--port", type=int, default=8080, help="Companion API 端口")
    parser.add_argument(
        "--discovery-timeout", type=float, default=10.0, help="自动发现超时（秒）"
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=STREAM_CHOICES,
        default=["gaze", "imu"],
        help="要接收的数据流，可同时指定多个",
    )
    parser.add_argument(
        "--imu-mode",
        choices=("direct", "simple"),
        default="direct",
        help=(
            "direct 直接读取异步 RTSP 并生成 Export 兼容文件；"
            "simple 使用官方单元素缓存接口"
        ),
    )
    parser.add_argument("--output", type=Path, help="输出目录")
    parser.add_argument("--no-save", action="store_true", help="只实时接收，不写入磁盘")
    parser.add_argument("--duration", type=float, help="运行秒数；省略则持续运行")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help="每种结构化数据的终端打印间隔；0 表示不打印",
    )
    parser.add_argument(
        "--preview", action="store_true", help="显示带注视点的场景视频（按 q/Esc 停止）"
    )
    parser.add_argument("--video-fps", type=float, default=30.0, help="保存场景视频的帧率")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.port < 1 or args.port > 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    if args.discovery_timeout <= 0:
        parser.error("--discovery-timeout 必须大于 0")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if args.print_interval < 0:
        parser.error("--print-interval 不能小于 0")
    if args.video_fps <= 0:
        parser.error("--video-fps 必须大于 0")
    if args.preview and "scene" not in args.streams:
        parser.error("--preview 需要在 --streams 中包含 scene")
    if args.no_save and args.output is not None:
        parser.error("--no-save 和 --output 不能同时使用")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

