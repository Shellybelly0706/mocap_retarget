#!/usr/bin/env python3
"""Play aligned OptiTrack in MuJoCo beside Neon front video with one transport."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session", type=Path, help="derived session from tests/align_recording.py"
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        help="local licensed SMPLX_NEUTRAL.npz; otherwise show solved skeleton",
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
        help="validate mapping, video and compile scene without a GUI",
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    scene = RecordingScene(
        session, args.smplx_model, args.mount, use_imu=args.imu_orientation
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
    if args.focus_head:
        scene.focus_head()
    if args.export_scene:
        scene.save_xml(args.export_scene)
    print(
        f"{len(session.times)} paired frames, {session.times[-1]:.3f}s | {scene.label}"
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


if __name__ == "__main__":
    main()
