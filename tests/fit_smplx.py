#!/usr/bin/env python3
"""Fit raw OptiTrack markers with the existing MoSh++ Stage I/II solver."""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c3d",
        type=Path,
        default=root / "data/1126/mocap/c3d/Take 2026-11-26 11.50.55 AM.c3d",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=root
        / "assets/avatar/smplx/marker_layout/Zekun_upper_25/marker_layout.json",
    )
    parser.add_argument("--model-dir", type=Path, default=root / "assets/avatar/smplx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--subject-prefix", default="Zixuan_Skeleton")
    parser.add_argument("--subject-id", type=int, default=0)
    parser.add_argument("--source-up", choices=("y", "z"), default="z")
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--least-avail-markers", type=float, default=0.95)
    parser.add_argument("--optimize-fingers", action="store_true")
    parser.add_argument("--stagei-only", action="store_true")
    parser.add_argument(
        "--init-layout",
        action="store_true",
        help="initialize a draft using exact known labels; never overwrite a valid layout",
    )
    parser.add_argument(
        "--reference-layout",
        type=Path,
        help="reuse exact label assignments from the prior marker layout",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="write preflight diagnostics without starting optimization",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "src/moshpp/src"))
    from utils.smplx.pipeline import preflight, run_legacy_fit

    if args.reference_layout and not args.init_layout:
        parser.error(
            "--reference-layout requires --init-layout and a new --layout path"
        )
    if args.init_layout:
        from utils.markers.layout import initialize_layout
        from utils.markers.layout_io import load_c3d_labels
        from utils.paths import MOSHPP_DIR

        labels = load_c3d_labels(
            args.c3d, MOSHPP_DIR, marker_prefix=args.subject_prefix
        )
        initialize_layout(
            args.layout,
            labels,
            args.c3d,
            MOSHPP_DIR,
            reference_layout_path=args.reference_layout,
        )
    output = args.output or root / "data/1126/reconstruction/recording_1" / args.mode
    if args.check_only:
        report = preflight(
            args.c3d,
            args.layout,
            model_dir=args.model_dir,
            subject_prefix=args.subject_prefix,
            mode=args.mode,
            sample_step=args.sample_step,
            least_avail_markers=args.least_avail_markers,
        )
        report["source_up_axis"] = args.source_up
        output.mkdir(parents=True, exist_ok=True)
        target = output / "preflight_report.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"MoSh++ {args.mode}: {report['expected_fitted_frames']} frames at {report['output_frame_rate']:g} Hz"
        )
        print(
            f"Marker layout: {len(report['assigned_markers'])}/{report['layout_marker_count']} assigned"
        )
        for blocker in report["blockers"]:
            print("Needs:", blocker)
        print("Saved:", target)
        return 0 if report["ready"] else 2
    report = run_legacy_fit(
        args.c3d,
        args.layout,
        output,
        model_dir=args.model_dir,
        subject_prefix=args.subject_prefix,
        subject_id=args.subject_id,
        source_up=args.source_up,
        mode=args.mode,
        sample_step=args.sample_step,
        optimize_fingers=args.optimize_fingers,
        least_avail_markers=args.least_avail_markers,
        stagei_only=args.stagei_only,
    )
    print("Saved:", report.get("smplx_parameters", report["stagei_result"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
