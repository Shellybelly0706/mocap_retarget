#!/usr/bin/env python3
"""Visualize an OptiTrack C3D or Motive CSV capture in MuJoCo."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="path to a .c3d or .csv capture")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument(
        "--no-loop", action="store_true", help="stop on the final frame"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="validate without opening a GUI"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from utils.optitrack.visualization import visualize_capture

        visualize_capture(
            args.capture,
            speed=args.speed,
            no_loop=args.no_loop,
            check_only=args.check_only,
        )
    except ModuleNotFoundError as error:
        parser.exit(
            2,
            f"Missing dependency {error.name!r}. Activate the environment from "
            "environment.yml.\n",
        )

    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
