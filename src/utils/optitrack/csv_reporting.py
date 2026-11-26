"""Reusable optitrack csv reporting helpers."""

from __future__ import annotations

import numpy as np

from utils.optitrack.csv import (
    CsvExport,
)
from utils.optitrack.csv_scene import (
    make_skeleton_entities,
)


def print_summary(csv_export: CsvExport) -> None:
    print(f"CSV: {csv_export.path}")
    print(
        f"Frames: {len(csv_export.times)} @ {csv_export.frame_rate:g} Hz; "
        f"duration: {csv_export.times[-1] - csv_export.times[0]:.3f} s"
    )
    skeleton_links = make_skeleton_entities(csv_export)
    print(
        f"Bone: {len(csv_export.bone_names)} points; "
        f"Skeleton: {len(skeleton_links)} parent-child links"
    )
    for category, (names, values) in csv_export.point_groups.items():
        valid = np.isfinite(values).all(axis=2)
        coverage = 100.0 * valid.mean() if valid.size else 0.0
        print(
            f"{category}: {len(names)} tracks; "
            f"visible samples={valid.sum()}/{valid.size} ({coverage:.1f}%)"
        )
    print(f"Rigid Body: {len(csv_export.rigid_names)}")
    print(
        "Colors: Bone/Skeleton=blue, Bone Marker=green, Rigid Body=red, "
        "Rigid Body Marker=orange, Marker=gray"
    )
