"""Reusable smplx playback helpers."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from utils.smplx.models import (
    StaticScene,
)


def effective_frame_rate(result: np.lib.npyio.NpzFile, frame_count: int) -> float:
    stored_fps = float(np.asarray(result["mocap_frame_rate"]).reshape(()))
    if stored_fps <= 0:
        raise ValueError(f"Invalid mocap_frame_rate: {stored_fps}")
    if "mocap_time_length" not in result.files:
        return stored_fps
    duration = float(np.asarray(result["mocap_time_length"]).reshape(()))
    inferred_fps = frame_count / duration if duration > 0 else stored_fps
    return inferred_fps if abs(stored_fps - inferred_fps) > 1.0 else stored_fps


def mocap_ids(model: mujoco.MjModel, joint_count: int) -> np.ndarray:
    ids = np.empty(joint_count, dtype=np.int32)
    for joint in range(joint_count):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"smplx_joint_{joint:02d}"
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body for SMPL-X joint {joint} was not compiled")
        ids[joint] = model.body_mocapid[body_id]
    if np.any(ids < 0):
        raise RuntimeError("One or more SMPL-X joint bodies are not mocap bodies")
    return ids


def cup_mocap_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "massless_cup")
    if body_id < 0 or model.body_mocapid[body_id] < 0:
        raise RuntimeError("The weightless cup was not compiled as a mocap body")
    return int(model.body_mocapid[body_id])


def set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    cup_id: int,
    cup_positions: np.ndarray,
    cup_quaternions: np.ndarray,
    frame: int,
) -> None:
    data.mocap_pos[ids] = positions[frame]
    data.mocap_quat[ids] = quaternions[frame]
    data.mocap_pos[cup_id] = cup_positions[frame]
    data.mocap_quat[cup_id] = cup_quaternions[frame]
    data.time = frame * model.opt.timestep
    mujoco.mj_forward(model, data)


def run_viewer(
    model: mujoco.MjModel,
    positions: np.ndarray,
    quaternions: np.ndarray,
    scene: StaticScene,
    start_frame: int,
    speed: float,
    loop: bool,
) -> None:
    data = mujoco.MjData(model)
    ids = mocap_ids(model, positions.shape[1])
    cup_id = cup_mocap_id(model)
    frame_count = len(positions)
    state = {"frame": start_frame, "paused": False, "loop": loop, "step": 0}

    def key_callback(keycode: int) -> None:
        if keycode == 32:  # Space
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):  # R/r
            state["frame"] = start_frame
            state["step"] = 0
        elif keycode == 262:  # Right arrow
            state["paused"] = True
            state["step"] = 1
        elif keycode == 263:  # Left arrow
            state["paused"] = True
            state["step"] = -1
        elif keycode in (76, 108):  # L/l
            state["loop"] = not state["loop"]

    set_frame(
        model,
        data,
        ids,
        positions,
        quaternions,
        cup_id,
        scene.cup_positions,
        scene.cup_quaternions,
        state["frame"],
    )
    frame_duration = model.opt.timestep / speed
    accumulator = 0.0
    previous_time = time.monotonic()

    print("Controls: Space pause/resume | Left/Right step | R restart | L toggle loop")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=True
    ) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.35, 1.15)
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 20.0
        viewer.cam.elevation = -8.0
        displayed_overlay: tuple[int, bool] | None = None

        while viewer.is_running():
            now = time.monotonic()
            elapsed = min(now - previous_time, 0.1)
            previous_time = now

            if state["step"]:
                state["frame"] = int(
                    np.clip(state["frame"] + state["step"], 0, frame_count - 1)
                )
                state["step"] = 0
                accumulator = 0.0
            elif not state["paused"]:
                accumulator += elapsed
                if accumulator >= frame_duration:
                    advance = int(accumulator / frame_duration)
                    accumulator -= advance * frame_duration
                    next_frame = state["frame"] + advance
                    if next_frame >= frame_count:
                        if state["loop"]:
                            loop_length = frame_count - start_frame
                            next_frame = start_frame + (
                                (next_frame - start_frame) % loop_length
                            )
                        else:
                            next_frame = frame_count - 1
                            state["paused"] = True
                    state["frame"] = next_frame

            set_frame(
                model,
                data,
                ids,
                positions,
                quaternions,
                cup_id,
                scene.cup_positions,
                scene.cup_quaternions,
                state["frame"],
            )
            overlay_state = (state["frame"], state["paused"])
            if overlay_state != displayed_overlay:
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Frame (0-based)\nPlayback",
                        f"{state['frame']} / {frame_count - 1}\n"
                        f"{'Paused' if state['paused'] else 'Running'}",
                    )
                )
                displayed_overlay = overlay_state
            viewer.sync()
            time.sleep(0.001)
