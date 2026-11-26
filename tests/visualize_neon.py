#!/usr/bin/env python3
"""Inspect the Neon glasses CAD and factory-calibrated camera in MuJoCo."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="compile and validate without a GUI"
    )
    parser.add_argument("--camera", choices=("orbit", "scene"), default="orbit")
    parser.add_argument(
        "--render", type=Path, help="save a PNG instead of opening the viewer"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.neon.viewer import render_neon, validate_neon_model, view_neon

    report = validate_neon_model()
    print(
        f"Neon {report['serial']}: {report['mesh_count']} meshes, {report['triangles']} triangles"
    )
    print(
        f"Scene camera: {report['resolution'][0]}x{report['resolution'][1]}; factory intrinsics"
    )
    print(
        "Camera pose: CAD lens-opening reference, not a calibrated optical-centre transform"
    )
    if args.render:
        print(f"Saved: {render_neon(args.render, camera=args.camera)}")
    elif not args.check_only:
        view_neon(camera=args.camera)


if __name__ == "__main__":
    main()
