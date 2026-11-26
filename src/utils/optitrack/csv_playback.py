"""Reusable optitrack csv playback helpers."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from utils.optitrack.csv import (
    available_takes,
    load_csv_export,
    resolve_take,
)
from utils.optitrack.csv_reporting import (
    print_summary,
)
from utils.optitrack.csv_scene import (
    CATEGORY_KEYS,
    build_model,
    make_entities,
    update_scene,
)


def visualize(args: argparse.Namespace) -> None:
    if args.list:
        for index, path in enumerate(available_takes(args.csv_dir), start=1):
            print(f"{index}: {path.name}")
        return

    csv_path = resolve_take(args.csv_dir, args.take)
    csv_export = load_csv_export(csv_path)
    print_summary(csv_export)

    entities = make_entities(csv_export)
    model = build_model(entities, 1.0 / csv_export.frame_rate)
    data = mujoco.MjData(model)
    visible = {category: True for category in CATEGORY_KEYS.values()}
    update_scene(data, entities, 0, visible)
    mujoco.mj_forward(model, data)
    print(f"MuJoCo mocap bodies: {model.nmocap}")
    if args.check_only:
        print("Validation complete; GUI was not opened (--check-only).")
        return

    from mujoco import viewer as mjviewer

    state = {"paused": False, "reset": False, "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):
            state["reset"] = True
        elif keycode in CATEGORY_KEYS:
            category = CATEGORY_KEYS[keycode]
            visible[category] = not visible[category]
            print(f"{category}: {'shown' if visible[category] else 'hidden'}")
        elif keycode == 262:
            state["paused"] = True
            state["step"] = 1
        elif keycode == 263:
            state["paused"] = True
            state["step"] = -1

    frame = 0
    fractional_frames = 0.0
    previous_time = time.perf_counter()
    finite_bones = csv_export.bone_positions[
        np.isfinite(csv_export.bone_positions).all(axis=2)
    ]
    center = (
        np.median(finite_bones, axis=0)
        if len(finite_bones)
        else np.array([0.0, 0.0, 1.0])
    )
    span = float(np.ptp(finite_bones, axis=0).max()) if len(finite_bones) else 2.0

    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = max(3.5, span * 1.5)
        viewer.cam.azimuth = 115
        viewer.cam.elevation = -18
        print(
            "Controls: Space pause/resume | R restart | Left/Right step | "
            "1 Bone | 2 Bone Marker | 3 Rigid Body | "
            "4 Rigid Body Marker | 5 Marker | 6 Skeleton | Esc close"
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
                    np.clip(frame + state["step"], 0, len(csv_export.times) - 1)
                )
                state["step"] = 0
            elif not state["paused"]:
                fractional_frames += elapsed * csv_export.frame_rate * args.speed
                advance = int(fractional_frames)
                fractional_frames -= advance
                if advance:
                    frame += advance
                    if frame >= len(csv_export.times):
                        if args.no_loop:
                            frame = len(csv_export.times) - 1
                            state["paused"] = True
                        else:
                            frame %= len(csv_export.times)

            update_scene(data, entities, frame, visible)
            data.time = csv_export.times[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.001)
