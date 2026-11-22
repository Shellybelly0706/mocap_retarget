#!/usr/bin/env python
"""Render a MoSh++ SMPL-X NPZ to an MP4 without OpenGL dependencies."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel


ROOT = Path("/media/fiona/LinuxData/mocap_retarget")
DEFAULT_RESULT = (
    ROOT
    / "moshpp_outputs/Zekun_upper_1_full/1112/c3d/Skeleton_Zixuan"
    / "Zekun_upper_1_stageii.npz"
)
DEFAULT_MODEL = ROOT / "mocap_data/data/smplx/SMPLX_NEUTRAL.npz"


def as_tensor(value):
    return torch.tensor(np.asarray(value), dtype=torch.float32)


def reconstruct_vertices(data, model_path, frame_ids, chunk_size=32):
    model = BodyModel(
        bm_fname=str(model_path),
        num_betas=int(data["num_betas"]),
        model_type="smplx",
    )
    faces = model.f.detach().cpu().numpy().astype(np.int32)
    vertices = None

    with torch.no_grad():
        for start in range(0, len(frame_ids), chunk_size):
            ids = frame_ids[start:start + chunk_size]
            count = len(ids)
            body = model(
                root_orient=as_tensor(data["root_orient"][ids]),
                pose_body=as_tensor(data["pose_body"][ids]),
                pose_hand=as_tensor(data["pose_hand"][ids]),
                pose_jaw=as_tensor(data["pose_jaw"][ids]),
                pose_eye=as_tensor(data["pose_eye"][ids]),
                betas=as_tensor(np.repeat(data["betas"][None], count, axis=0)),
                trans=as_tensor(data["trans"][ids]),
            )
            chunk = body.v.detach().cpu().numpy()
            if vertices is None:
                vertices = np.empty(
                    (len(frame_ids), chunk.shape[1], 3), dtype=np.float32
                )
            vertices[start:start + count] = chunk

    return vertices, faces


def projection_bounds(vertices, horizontal_index, panel_width, panel_height, margin):
    horizontal = vertices[:, :, horizontal_index]
    vertical = vertices[:, :, 1]
    hmin, hmax = float(horizontal.min()), float(horizontal.max())
    vmin, vmax = float(vertical.min()), float(vertical.max())
    hpad = max((hmax - hmin) * 0.08, 0.03)
    vpad = max((vmax - vmin) * 0.05, 0.03)
    hmin, hmax = hmin - hpad, hmax + hpad
    vmin, vmax = vmin - vpad, vmax + vpad
    scale = min(
        (panel_width - 2 * margin) / (hmax - hmin),
        (panel_height - 2 * margin) / (vmax - vmin),
    )
    used_width = (hmax - hmin) * scale
    used_height = (vmax - vmin) * scale
    x_offset = (panel_width - used_width) * 0.5
    y_offset = (panel_height - used_height) * 0.5
    return hmin, vmin, scale, x_offset, y_offset


def project(points, horizontal_index, bounds, panel_height, mirror=False):
    hmin, vmin, scale, x_offset, y_offset = bounds
    horizontal = points[:, horizontal_index]
    if mirror:
        horizontal = -horizontal
        hmin = -hmin
    x = x_offset + (horizontal - hmin) * scale
    y = panel_height - (y_offset + (points[:, 1] - vmin) * scale)
    return np.column_stack((x, y)).round().astype(np.int32)


def render_panel(vertices, faces, markers_obs, markers_sim, horizontal_index,
                 depth_index, bounds, width, height, title):
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    points_2d = project(vertices, horizontal_index, bounds, height)
    triangles = points_2d[faces]
    face_depth = vertices[faces, depth_index].mean(axis=1)

    # Approximate painter's algorithm using depth bands. The camera looks from
    # the positive depth axis, so small-depth faces are drawn first.
    depth_min, depth_max = float(face_depth.min()), float(face_depth.max())
    edges = np.linspace(depth_min, depth_max + 1e-9, 33)
    for index in range(len(edges) - 1):
        mask = (face_depth >= edges[index]) & (face_depth < edges[index + 1])
        if not np.any(mask):
            continue
        shade = int(150 + 75 * index / (len(edges) - 2))
        cv2.fillPoly(panel, triangles[mask], (shade, shade, min(shade + 8, 255)),
                     lineType=cv2.LINE_AA)

    cv2.polylines(panel, triangles[::64], True, (115, 115, 125), 1,
                  lineType=cv2.LINE_AA)

    obs_2d = project(markers_obs, horizontal_index, bounds, height)
    sim_2d = project(markers_sim, horizontal_index, bounds, height)
    for point in obs_2d:
        cv2.circle(panel, tuple(point), 5, (45, 45, 230), -1, cv2.LINE_AA)
    for point in sim_2d:
        cv2.circle(panel, tuple(point), 3, (45, 200, 70), -1, cv2.LINE_AA)

    cv2.putText(panel, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (35, 35, 35), 2, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    result_path = args.result.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else result_path.with_name(result_path.stem + "_preview.mp4")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(str(result_path), allow_pickle=True)
    total_frames = len(data["trans"])
    frame_ids = np.arange(0, total_frames, args.stride, dtype=np.int64)
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    stored_fps = float(data["mocap_frame_rate"])
    stored_duration = float(data["mocap_time_length"])
    inferred_fps = total_frames / stored_duration if stored_duration > 0 else stored_fps
    # Older MoSh++ exports retain the source rate after downsampling.
    effective_fps = inferred_fps if abs(stored_fps - inferred_fps) > 1.0 else stored_fps
    video_fps = effective_fps / args.stride

    print("Reconstructing {} of {} frames...".format(len(frame_ids), total_frames))
    vertices, faces = reconstruct_vertices(data, args.model.resolve(), frame_ids)
    markers_obs = np.asarray(data["markers_obs"])[frame_ids]
    markers_sim = np.asarray(data["markers_sim"])[frame_ids]

    panel_width = args.width // 2
    front_bounds = projection_bounds(vertices, 2, panel_width, args.height, 35)
    side_bounds = projection_bounds(vertices, 0, panel_width, args.height, 35)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (panel_width * 2, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create video: {}".format(output_path))

    print("Rendering at {:.3f} fps...".format(video_fps))
    for output_index, source_index in enumerate(frame_ids):
        front = render_panel(
            vertices[output_index], faces, markers_obs[output_index],
            markers_sim[output_index], 2, 0, front_bounds,
            panel_width, args.height, "Front view",
        )
        side = render_panel(
            vertices[output_index], faces, markers_obs[output_index],
            markers_sim[output_index], 0, 2, side_bounds,
            panel_width, args.height, "Side view",
        )
        frame = np.concatenate((front, side), axis=1)
        cv2.putText(
            frame,
            "frame {}/{}   time {:.2f}s".format(
                int(source_index), total_frames - 1, source_index / effective_fps
            ),
            (18, args.height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (30, 30, 30), 2, cv2.LINE_AA,
        )
        cv2.circle(frame, (22, args.height - 22), 5, (45, 45, 230), -1)
        cv2.putText(frame, "observed marker", (34, args.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 120), 1, cv2.LINE_AA)
        cv2.circle(frame, (180, args.height - 22), 4, (45, 200, 70), -1)
        cv2.putText(frame, "fitted marker", (191, args.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 120, 45), 1, cv2.LINE_AA)
        writer.write(frame)

        if (output_index + 1) % 50 == 0 or output_index + 1 == len(frame_ids):
            print("Rendered {}/{}".format(output_index + 1, len(frame_ids)))

    writer.release()
    print("Saved:", output_path)
    print("Playback rate: {:.3f} fps".format(video_fps))


if __name__ == "__main__":
    main()
