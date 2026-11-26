"""Reusable synchronization reporting helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from utils.synchronization.clocks import (
    utc_iso,
)
from utils.synchronization.models import (
    AffineMap,
    ClockAnchor,
    EventAnchor,
    MotiveExport,
)


def write_outputs(
    output_csv: Path,
    report_path: Path,
    motive: MotiveExport,
    clock_anchors: Sequence[ClockAnchor],
    events: Sequence[EventAnchor],
    model: AffineMap,
    ubuntu_to_companion: Callable[[int], int],
    args: argparse.Namespace,
    automatic_diagnostics: dict[str, object] | None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    first_ubuntu = model.ubuntu_ns(0.0)
    last_elapsed = (motive.frame_ids[-1] - motive.first_frame) / motive.frame_rate_hz
    last_ubuntu = model.ubuntu_ns(last_elapsed)

    # Validate clock-log coverage before creating a partial CSV.
    first_companion = ubuntu_to_companion(first_ubuntu)
    last_companion = ubuntu_to_companion(last_ubuntu)

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "motive_frame",
                "motive_elapsed_s",
                "ubuntu_unix_ns",
                "ubuntu_utc_iso",
                "estimated_neon_companion_unix_ns",
            ]
        )
        for frame in motive.frame_ids:
            elapsed_s = (frame - motive.first_frame) / motive.frame_rate_hz
            ubuntu_ns = model.ubuntu_ns(elapsed_s)
            companion_ns = ubuntu_to_companion(ubuntu_ns)
            writer.writerow(
                [
                    frame,
                    f"{elapsed_s:.9f}",
                    ubuntu_ns,
                    utc_iso(ubuntu_ns),
                    companion_ns,
                ]
            )

    residual_abs = [abs(value) for value in model.residuals_ms]
    report = {
        "schema_version": 1,
        "purpose": "Offline Motive to Ubuntu/Neon clock synchronization",
        "inputs": {
            "motive_csv": str(motive.path.resolve()),
            "clock_offsets_jsonl": str(args.offset_log.resolve()),
            "neon_time_file": (
                str(args.neon_time_file.resolve()) if args.neon_time_file else None
            ),
            "neon_imu_raw": (
                str(args.neon_imu_raw.resolve()) if args.neon_imu_raw else None
            ),
            "neon_imu_time": (
                str(args.neon_imu_time.resolve()) if args.neon_imu_time else None
            ),
        },
        "motive": {
            "take_name": motive.take_name,
            "frame_rate_hz": motive.frame_rate_hz,
            "first_exported_frame": motive.first_frame,
            "last_exported_frame": motive.frame_ids[-1],
            "frame_count": len(motive.frame_ids),
            "capture_start_frame_metadata": motive.capture_start_frame,
        },
        "time_echo": {
            "used_anchor_count": len(clock_anchors),
            "included_unstable_records": bool(args.include_unstable),
            "allowed_out_of_range_endpoint_offset": bool(
                args.allow_offset_extrapolation
            ),
            "companion_coverage_ns": [
                clock_anchors[0].companion_ns,
                clock_anchors[-1].companion_ns,
            ],
            "ubuntu_coverage_ns": [
                clock_anchors[0].ubuntu_ns,
                clock_anchors[-1].ubuntu_ns,
            ],
        },
        "event_anchors": [
            {
                "motive_frame": event.motive_frame,
                "motive_elapsed_s": event.motive_elapsed_s,
                "neon_companion_unix_ns": event.companion_ns,
                "ubuntu_unix_ns_after_time_echo": event.ubuntu_ns,
                "source": event.source,
                "fit_residual_ms": residual,
            }
            for event, residual in zip(events, model.residuals_ms)
        ],
        "automatic_alignment": automatic_diagnostics,
        "fitted_mapping": {
            "formula": ("ubuntu_ns = intercept_ns + slope * motive_elapsed_s * 1e9"),
            "slope": model.slope,
            "estimated_rate_drift_ppm": (model.slope - 1.0) * 1_000_000,
            "intercept_ns": model.intercept_ns,
            "one_anchor_assumes_zero_drift": len(events) == 1,
            "max_absolute_anchor_residual_ms": max(residual_abs),
            "rms_anchor_residual_ms": math.sqrt(
                sum(value * value for value in model.residuals_ms)
                / len(model.residuals_ms)
            ),
        },
        "mapped_range": {
            "first_ubuntu_unix_ns": first_ubuntu,
            "last_ubuntu_unix_ns": last_ubuntu,
            "first_ubuntu_utc": utc_iso(first_ubuntu),
            "last_ubuntu_utc": utc_iso(last_ubuntu),
            "first_estimated_neon_companion_unix_ns": first_companion,
            "last_estimated_neon_companion_unix_ns": last_companion,
        },
        "outputs": {"frame_timestamps_csv": str(output_csv.resolve())},
        "limitations": [
            "Accuracy is limited by how precisely the same physical event is selected in both recordings.",
            "Time Echo has millisecond timestamp resolution and does not guarantee sub-millisecond absolute accuracy.",
            "With one event anchor, clock-rate drift cannot be estimated.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
