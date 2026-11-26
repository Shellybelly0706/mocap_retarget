#!/usr/bin/env python3
"""Crop OptiTrack and native Neon recordings to their shared global timestamps."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument(
        "--neon", type=Path, required=True, help="native Neon recording directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new derived session directory"
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Berlin",
        help="Motive capture computer's wall-clock timezone",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.synchronization.preprocessing import align_recording

    report = align_recording(
        args.motive, args.neon, args.output, timezone=args.timezone
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "frame_count",
                    "duration_s",
                    "maximum_nearest_sample_delta_ms",
                    "clock_assumption",
                )
            },
            indent=2,
        )
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
