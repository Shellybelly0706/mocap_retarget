"""Reusable smplx rendering helpers."""

from __future__ import annotations

import cv2
import numpy as np


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


def render_panel(
    vertices,
    faces,
    markers_obs,
    markers_sim,
    horizontal_index,
    depth_index,
    bounds,
    width,
    height,
    title,
    show_markers=True,
):
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
        cv2.fillPoly(
            panel,
            triangles[mask],
            (shade, shade, min(shade + 8, 255)),
            lineType=cv2.LINE_AA,
        )

    cv2.polylines(
        panel, triangles[::64], True, (115, 115, 125), 1, lineType=cv2.LINE_AA
    )

    if show_markers:
        obs_2d = project(markers_obs, horizontal_index, bounds, height)
        sim_2d = project(markers_sim, horizontal_index, bounds, height)
        for point in obs_2d:
            cv2.circle(panel, tuple(point), 5, (45, 45, 230), -1, cv2.LINE_AA)
        for point in sim_2d:
            cv2.circle(panel, tuple(point), 3, (45, 200, 70), -1, cv2.LINE_AA)

    cv2.putText(
        panel,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    return panel
