"""Crop native Neon video and sample Motive rows onto its timestamp grid.

No motion-based clock correction or interpolation is applied. The output mapping
is authoritative: CSV Frame/Time cells retain their original source meanings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from .motive_motion import parse_motive_start_time


def nearest_samples(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    """Find nearest samples using integer nanoseconds; ties choose the earlier."""
    source = np.asarray(source_ns, dtype=np.int64)
    target = np.asarray(target_ns, dtype=np.int64)
    if not len(source) or np.any(np.diff(source) <= 0):
        raise ValueError("Source timestamps must be nonempty and strictly increasing")
    right = np.searchsorted(source, target).clip(0, len(source) - 1)
    left = (right - 1).clip(0)
    return np.where(target - source[left] <= source[right] - target, left, right)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_video(
    source: Path, output: Path, first: int, last: int, times_ns: np.ndarray
) -> None:
    """Decode exact frame indices and reencode, retaining variable frame timing."""
    import av

    written = 0
    with av.open(str(source)) as reader, av.open(str(output), mode="w") as writer:
        original = reader.streams.video[0]
        stream = writer.add_stream("libx264", rate=original.average_rate or 30)
        stream.width, stream.height = original.width, original.height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = stream.time_base
        stream.codec_context.thread_count = 4
        stream.options = {"crf": "20", "preset": "fast"}
        for index, frame in enumerate(reader.decode(video=0)):
            if index < first:
                continue
            if index > last:
                break
            frame.pts = int((int(times_ns[written]) - int(times_ns[0]) + 500) // 1000)
            frame.time_base = Fraction(1, 1_000_000)
            for packet in stream.encode(frame):
                writer.mux(packet)
            written += 1
        if written != len(times_ns):
            raise ValueError(
                f"Video ended early: expected {len(times_ns)}, decoded {written}"
            )
        for packet in stream.encode():
            writer.mux(packet)


def align_recording(
    motive_csv: Path,
    neon_dir: Path,
    output_dir: Path,
    *,
    timezone: str = "Europe/Berlin",
) -> dict:
    """Write a derived equal-count video/CSV pair in a fresh output directory."""
    motive_csv, neon_dir, output_dir = map(Path, (motive_csv, neon_dir, output_dir))
    videos = sorted(neon_dir.glob("Neon Scene Camera*.mp4"))
    if len(videos) != 1:
        raise ValueError(
            "Expected one native Neon scene video in the recording directory"
        )
    video = videos[0]
    time_file = video.with_suffix(".time")
    if time_file.stat().st_size % 8:
        raise ValueError("Neon .time file is not an int64 nanosecond sequence")
    neon_ns = np.fromfile(time_file, dtype="<i8")
    with motive_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = [next(reader) for _ in range(7)]
        rows = [row for row in reader if row and row[0].strip()]
    metadata = dict(zip(header[0][::2], header[0][1::2]))
    start = parse_motive_start_time(metadata["Capture Start Time"], timezone)
    motive_ns = start + np.array(
        [round(float(row[1]) * 1e9) for row in rows], dtype=np.int64
    )
    for name, stamps in (("Neon", neon_ns), ("Motive", motive_ns)):
        if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
            raise ValueError(
                f"{name} timestamps must contain at least two increasing samples"
            )
    import av

    with av.open(str(video)) as container:
        decoded_count = sum(1 for _ in container.decode(video=0))
    if decoded_count != len(neon_ns):
        raise ValueError(
            f"Video has {decoded_count} frames but .time has {len(neon_ns)} timestamps"
        )
    overlap = [
        max(int(neon_ns[0]), int(motive_ns[0])),
        min(int(neon_ns[-1]), int(motive_ns[-1])),
    ]
    selected = np.flatnonzero((neon_ns >= overlap[0]) & (neon_ns <= overlap[1]))
    if len(selected) < 2:
        raise ValueError("Recordings have no usable common global-time interval")
    times = neon_ns[selected]
    sampled = nearest_samples(motive_ns, times)
    delta_ms = (motive_ns[sampled] - times) / 1e6
    report = {
        "schema_version": 1,
        "method": "common global-time interval; nearest Motive sample per native Neon frame",
        "clock_assumption": "Motive wall clock and Neon Unix clock agree; not independently verified",
        "motive_timezone": timezone,
        "clock_correction_ns": 0,
        "interpolation": "none; source missing values retained",
        "canonical_stream": "Neon scene camera native .time timestamps",
        "coordinate_convention": "source Motive coordinates retained; 1126 capture is Z-up",
        "source_length_units": metadata.get("Length Units", "unknown"),
        "motive_metadata": metadata,
        "sources": {
            key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for key, path in (
                ("motive_csv", motive_csv),
                ("neon_video", video),
                ("neon_timestamps", time_file),
            )
        },
        "source_counts": {"motive": len(rows), "neon": len(neon_ns)},
        "source_ranges_ns": {
            "motive": [int(motive_ns[0]), int(motive_ns[-1])],
            "neon": [int(neon_ns[0]), int(neon_ns[-1])],
        },
        "overlap_ns": overlap,
        "retained_range_ns": [int(times[0]), int(times[-1])],
        "neon_source_frame_range": [int(selected[0]), int(selected[-1])],
        "frame_count": len(times),
        "duration_s": float((times[-1] - times[0]) / 1e9),
        "maximum_nearest_sample_delta_ms": float(np.abs(delta_ms).max()),
        "csv_convention": "Frame/Time and capture/export rates retain source meaning; irregular sampled rows. Use frame_alignment.csv for the aligned frame index/time.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        crop_video(
            video,
            output_dir / "front_camera_aligned.mp4",
            int(selected[0]),
            int(selected[-1]),
            times,
        )
        with (output_dir / "mocap_aligned.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            # Keep source clock/rates/IDs; update only the actual exported row count.
            for key in ("Total Frames in Take", "Total Exported Frames"):
                if key in header[0]:
                    header[0][header[0].index(key) + 1] = (
                        str(len(times))
                        if key == "Total Exported Frames"
                        else metadata[key]
                    )
            writer.writerows(header)
            writer.writerows(rows[index] for index in sampled)
        with (output_dir / "frame_alignment.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "aligned_frame",
                    "timestamp_ns",
                    "time_s",
                    "neon_source_frame",
                    "neon_timestamp_ns",
                    "mocap_source_row",
                    "mocap_source_frame",
                    "mocap_timestamp_ns",
                    "mocap_delta_ms",
                ]
            )
            for index, (neon_index, mocap_index) in enumerate(
                zip(selected, sampled, strict=True)
            ):
                writer.writerow(
                    [
                        index,
                        int(times[index]),
                        f"{(times[index] - times[0]) / 1e9:.9f}",
                        int(neon_index),
                        int(times[index]),
                        int(mocap_index),
                        rows[mocap_index][0],
                        int(motive_ns[mocap_index]),
                        f"{delta_ms[index]:.6f}",
                    ]
                )
        (output_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception:
        # Leave a conspicuous marker; never mistake a partial export for a session.
        (output_dir / "INCOMPLETE").write_text(
            "Alignment failed; remove this derived directory before retrying.\n"
        )
        raise
    return report
