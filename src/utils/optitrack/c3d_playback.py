"""Reusable optitrack c3d playback helpers."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from utils.optitrack.c3d import (
    load_c3d,
    print_summary,
)
from utils.optitrack.c3d_scene import (
    build_model,
    initialize_marker_geoms,
    initialize_skeleton_geoms,
    update_marker_geoms,
    update_skeleton_geoms,
)
from utils.optitrack.skeleton import (
    build_skeleton_motion,
    display_labels,
)


def visualize(args: argparse.Namespace) -> None:
    motion = load_c3d(args.c3d.expanduser().resolve(), args.c3d_reader.resolve())
    labels = display_labels(motion.labels, args.short_labels)
    skeleton = None if args.no_skeleton else build_skeleton_motion(motion)
    print_summary(motion)
    if skeleton is not None:
        print(
            f"Skeleton: {len(skeleton.node_names)} virtual joints, "
            f"{len(skeleton.edges)} connected bone segments"
        )
    if args.check_only:
        print("Validation complete; GUI was not opened (--check-only).")
        return

    finite_positions = motion.positions[motion.valid]
    center = np.median(finite_positions, axis=0)
    span = float(np.ptp(finite_positions, axis=0).max())
    floor_height = float(np.nanmin(motion.positions[:, :, 2]) - 0.05)
    model = build_model(1.0 / motion.frame_rate, floor_height)
    data = mujoco.MjData(model)

    from mujoco import viewer as mjviewer

    state = {
        "paused": False,
        "reset": False,
        "step": 0,
        "labels": True,
        "markers": True,
        "skeleton": skeleton is not None,
    }

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):
            state["reset"] = True
        elif keycode in (76, 108):
            state["labels"] = not state["labels"]
            print(f"Labels: {'shown' if state['labels'] else 'hidden'}")
        elif keycode in (77, 109):
            state["markers"] = not state["markers"]
            print(f"Markers: {'shown' if state['markers'] else 'hidden'}")
        elif keycode in (66, 98) and skeleton is not None:
            state["skeleton"] = not state["skeleton"]
            print(f"Skeleton: {'shown' if state['skeleton'] else 'hidden'}")
        elif keycode == 262:
            state["paused"] = True
            state["step"] = 1
        elif keycode == 263:
            state["paused"] = True
            state["step"] = -1

    frame = 0
    fractional_frames = 0.0
    previous_time = time.perf_counter()
    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(2.0, span * 2.0)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        initialize_marker_geoms(viewer.user_scn, labels, args.marker_size)
        if skeleton is not None:
            initialize_skeleton_geoms(
                viewer.user_scn,
                len(labels),
                skeleton,
                args.joint_size,
                args.bone_width,
            )
        print(
            "Controls: Space pause/resume | R restart | Left/Right step | "
            "L labels | M markers | B skeleton | Esc close"
        )
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = min(now - previous_time, 0.1)
            previous_time = now
            if state["reset"]:
                frame = 0
                fractional_frames = 0.0
                state["reset"] = False
            if state["step"]:
                frame = int(
                    np.clip(frame + state["step"], 0, len(motion.positions) - 1)
                )
                state["step"] = 0
            elif not state["paused"]:
                fractional_frames += elapsed * motion.frame_rate * args.speed
                advance = int(fractional_frames)
                fractional_frames -= advance
                if advance:
                    frame += advance
                    if frame >= len(motion.positions):
                        if args.no_loop:
                            frame = len(motion.positions) - 1
                            state["paused"] = True
                        else:
                            frame %= len(motion.positions)
            update_marker_geoms(
                viewer.user_scn,
                motion,
                labels,
                frame,
                state["labels"],
                state["markers"],
            )
            if skeleton is not None:
                update_skeleton_geoms(
                    viewer.user_scn,
                    len(labels),
                    skeleton,
                    frame,
                    state["skeleton"],
                    args.bone_width,
                )
            data.time = frame / motion.frame_rate
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.001)
