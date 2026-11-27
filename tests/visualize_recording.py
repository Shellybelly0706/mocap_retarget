#!/usr/bin/env python3
"""Play aligned OptiTrack in MuJoCo beside Neon front video with one transport."""

import argparse
import json
import sys
from pathlib import Path


def infer_c3d(session) -> Path:
    """Locate the C3D corresponding to the aligned session's source CSV."""
    try:
        csv_path = Path(session.report["sources"]["motive_csv"]["path"])
    except KeyError as error:
        raise ValueError(
            "alignment.json does not identify its source Motive CSV; pass --fit-c3d"
        ) from error
    candidate = csv_path.parent.parent / "c3d" / f"{csv_path.stem}.c3d"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Could not infer the matching C3D at {candidate}; pass --fit-c3d"
        )
    return candidate


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "src/moshpp/src"))
    from utils.paths import DATA_DIR, SMPLX_ASSETS_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session", type=Path, help="derived session from tests/align_recording.py"
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        help="local licensed SMPLX_NEUTRAL.npz; otherwise show solved skeleton",
    )
    parser.add_argument(
        "--skeleton-source",
        choices=("motive", "smplx"),
        default="motive",
        help="show the exported Motive skeleton or the avatar's SMPL-X hierarchy",
    )
    result_group = parser.add_mutually_exclusive_group()
    result_group.add_argument(
        "--fit-smplx",
        action="store_true",
        help="run legacy MoSh++ Stage I/II, then show the fitted avatar",
    )
    result_group.add_argument(
        "--smplx-result",
        type=Path,
        help="existing Stage-II AMASS-style NPZ with source_frame_numbers",
    )
    parser.add_argument(
        "--fit-c3d",
        type=Path,
        help="raw C3D for --fit-smplx; inferred from alignment.json when omitted",
    )
    parser.add_argument(
        "--fit-layout",
        type=Path,
        default=SMPLX_ASSETS_DIR
        / "marker_layout/Zekun_upper_25/marker_layout.json",
        help="complete marker-to-SMPL-X layout used by MoSh++",
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=DATA_DIR / "1126/reconstruction/recording_1/full",
        help="directory for Stage I, Stage II, parameters and fit provenance",
    )
    parser.add_argument("--mount", type=Path, help="saved visual glasses mount JSON")
    parser.add_argument(
        "--imu-orientation",
        action="store_true",
        help="orient glasses using recorded module IMU after validating world-heading registration; nose position stays geometric",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate without a GUI; with --fit-smplx, write preflight only",
    )
    parser.add_argument(
        "--render",
        type=Path,
        help="save a paired preview PNG instead of opening the player",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--focus-head", action="store_true")
    parser.add_argument(
        "--export-scene",
        type=Path,
        help="write assembled avatar/glasses MJCF for inspection",
    )
    args = parser.parse_args()
    import cv2

    from utils.recording.scene import RecordingScene
    from utils.recording.session import load_session

    session = load_session(args.session)
    capture = cv2.VideoCapture(str(session.directory / "front_camera_aligned.mp4"))
    try:
        if not capture.isOpened() or round(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ) != len(session.times):
            raise ValueError("Aligned video and motion frame counts differ")
    finally:
        capture.release()

    default_fitted_model = SMPLX_ASSETS_DIR / "SMPLX_NEUTRAL.npz"
    fitted_result = args.smplx_result
    if args.fit_smplx:
        if (
            args.smplx_model
            and args.smplx_model.resolve() != default_fitted_model.resolve()
        ):
            parser.error(
                "--fit-smplx currently uses the neutral MoSh++ model; do not apply "
                "its fitted betas to a female or male template"
            )
        args.smplx_model = default_fitted_model
        c3d = args.fit_c3d or infer_c3d(session)
        from utils.smplx.pipeline import preflight, run_legacy_fit

        if args.check_only:
            report = preflight(
                c3d,
                args.fit_layout,
                subject_prefix="Zixuan_Skeleton",
                mode="full",
                # The authoritative paired mapping selects native Motive rows
                # that are not a perfectly regular every-fourth sequence.
                # Fit all C3D frames so playback never invents missing poses.
                sample_step=1,
                least_avail_markers=0.95,
            )
            report["source_up_axis"] = "z"
            args.fit_output.mkdir(parents=True, exist_ok=True)
            target = args.fit_output / "preflight_report.json"
            target.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"MoSh++ full: {report['expected_fitted_frames']} frames at "
                f"{report['output_frame_rate']:g} Hz"
            )
            for blocker in report["blockers"]:
                print("Needs:", blocker)
            print("Saved:", target)
            return 0 if report["ready"] else 2
        report = run_legacy_fit(
            c3d,
            args.fit_layout,
            args.fit_output,
            subject_prefix="Zixuan_Skeleton",
            source_up="z",
            mode="full",
            sample_step=1,
            optimize_fingers=False,
            least_avail_markers=0.95,
        )
        fitted_result = Path(report["smplx_parameters"])
        print("MoSh++ Stage I:", report["stagei_result"])
        print("MoSh++ Stage II:", report["stageii_result"])
        print("SMPL-X parameters:", fitted_result)
    elif fitted_result and not args.smplx_model:
        args.smplx_model = default_fitted_model

    if (
        args.fit_c3d
        or args.fit_output
        != DATA_DIR / "1126/reconstruction/recording_1/full"
    ) and not args.fit_smplx:
        parser.error("--fit-c3d and --fit-output require --fit-smplx")
    if args.skeleton_source == "smplx" and not args.smplx_model:
        parser.error("--skeleton-source smplx requires an SMPL-X model or result")
    scene = RecordingScene(
        session,
        args.smplx_model,
        args.mount,
        use_imu=args.imu_orientation,
        fitted_result_path=fitted_result,
        skeleton_source=args.skeleton_source,
    )
    if args.imu_orientation:
        import numpy as np

        from utils.neon.imu import save_imu_report

        save_imu_report(
            session.directory,
            scene.imu_sampled,
            scene.imu_registration,
            np.array(
                [int(row["timestamp_ns"]) for row in session.mapping], dtype=np.int64
            ),
        )
        registration = scene.imu_registration
        print(
            f"IMU world yaw registration: {registration['world_yaw_degrees']:.2f}°; median residual {registration['median_residual_degrees']:.2f}°; held-out p90 {registration['heldout']['p90_residual_degrees']:.2f}°"
        )
    if not 0 <= args.frame < len(session.times):
        parser.error("--frame is outside the aligned recording")
    scene.set_frame(args.frame)
    if scene.temple_fit:
        for side in ("left", "right"):
            fit = scene.temple_fit[side]
            status = (
                "clearance reached"
                if fit["clearance_reached"]
                else "clearance NOT reached"
            )
            print(
                f"Visual {side} arm opening: {fit['opening_deg']:.2f}°; approximate {status}"
            )
    if args.focus_head:
        scene.focus_head()
    if args.export_scene:
        scene.save_xml(args.export_scene)
    print(
        f"{len(session.times)} paired frames, {session.times[-1]:.3f}s | {scene.label}"
    )
    print(f"Skeleton: {scene.skeleton_label}")
    if fitted_result:
        registration = scene.fitted_registration
        print(
            "MoSh++ C3D -> paired Motive CSV registration: "
            f"median {registration['median_error_m']*1000:.3f} mm, "
            f"p95 {registration['p95_error_m']*1000:.3f} mm"
        )
    print(
        "Neon 370641 factory intrinsics; camera optical axes: X right/red, Y down/green, Z forward/blue"
    )
    print("Glasses placement is provisional; global clock agreement is unverified")
    print(
        "IMU frame: module top bar, X right / Y forward / Z up; nominal camera-relative origin"
    )
    if args.render:
        from utils.recording.rendering import render_preview

        render_preview(scene, args.render, args.frame)
        print(f"Saved {args.render}")
    elif not args.check_only:
        from utils.recording.player import RecordingPlayer

        player = RecordingPlayer(scene)
        player.seek(args.frame)
        player.run()
    if scene.temple_fit:
        scene.save_visual_fit(session.directory / "neon_visual_fit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
