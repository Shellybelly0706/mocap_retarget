"""Reusable smplx fitting helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT
from utils.paths import SMPLX_ASSETS_DIR

ROOT = _REPOSITORY_ROOT


MODEL_DIR = SMPLX_ASSETS_DIR


SEQUENCES = {
    "upper": {
        "c3d": ROOT / "data/1112/c3d/Zekun_upper_1.c3d",
        "marker_layout": MODEL_DIR / "marker_layout/Zekun_upper_25/marker_layout.json",
        "subject_id": 0,
        "least_avail_markers": 1.0,
        "optimize_fingers": False,
        "output_suffix": "",
        "output_tag": "layout_fixed_v2",
    },
    "righthand": {
        "c3d": ROOT / "data/1112/c3d/Zekun_upper_righthand_1.c3d",
        "marker_layout": MODEL_DIR
        / "marker_layout/Zekun_upper_righthand_32/marker_layout.json",
        # Body and hand markers live in separate Motive namespaces. -1 keeps both.
        "subject_id": -1,
        # Three wrist/hand labels occur in both namespaces, so the C3D contains
        # 35 tracks for 32 unique labels. No frame has all 35 tracks visible.
        "least_avail_markers": 0.95,
        "optimize_fingers": True,
        "output_suffix": "_fingers",
        "output_tag": "layout_fixed_v2",
    },
    "righthand-added": {
        "c3d": ROOT / "data/1112/c3d/Zekun_upper_righthand_added_1.c3d",
        "marker_layout": MODEL_DIR
        / "marker_layout/Zekun_upper_righthand_added_32/marker_layout.json",
        "subject_id": 0,
        "least_avail_markers": 1.0,
        "optimize_fingers": True,
        "output_suffix": "_fingers",
        "output_tag": "layout_fixed_v2",
    },
}


def load_layout_indices(marker_layout):
    """Read fitting assignments, preserving valid legacy skin offsets.

    The interactive surface editor requires zero offsets, whereas existing
    calibrated MoSh++ layouts can include nonzero marker-to-skin distances.
    """
    with Path(marker_layout).open("r", encoding="utf-8") as handle:
        layout = json.load(handle)
    if layout.get("surface_model_type") != "smplx":
        raise ValueError("Fitting layout must use the SMPL-X surface model")
    indices = {}
    for marker_set in layout.get("markersets", []):
        distance = marker_set.get("distance_from_skin", 0.0)
        if not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise ValueError("Marker-to-skin distance must be finite")
        for label, vertex_id in marker_set.get("indices", {}).items():
            if label in indices:
                raise ValueError(f"Marker layout contains duplicate label: {label}")
            if vertex_id is not None and (
                isinstance(vertex_id, bool) or not isinstance(vertex_id, int)
            ):
                raise ValueError(f"Invalid vertex ID for marker {label!r}")
            indices[label] = vertex_id
    return indices


def load_layout_markers(marker_layout):
    """Return complete marker assignments for the legacy fitting solver."""
    indices = load_layout_indices(marker_layout)
    markers = [label for label, vertex in indices.items() if vertex is not None]
    unassigned = [label for label, vertex in indices.items() if vertex is None]

    if unassigned:
        raise ValueError(
            "Marker layout contains unassigned labels: {}".format(
                ", ".join(sorted(unassigned))
            )
        )
    if not markers:
        raise ValueError(
            "Marker layout contains no assigned markers: {}".format(marker_layout)
        )
    if len(markers) != len(set(markers)):
        raise ValueError(
            "Marker layout contains duplicate labels: {}".format(marker_layout)
        )
    return markers


def build_overrides(mode, sequence):
    if mode not in {"smoke", "full"}:
        raise ValueError("Fitting mode must be smoke or full")
    sequence_cfg = SEQUENCES[sequence]
    c3d = sequence_cfg["c3d"]
    marker_layout = sequence_cfg["marker_layout"]
    output_name = "{}{}_{}_{}".format(
        c3d.stem, sequence_cfg["output_suffix"], sequence_cfg["output_tag"], mode
    )

    return build_capture_overrides(
        c3d,
        marker_layout,
        ROOT / "moshpp_outputs" / output_name,
        subject_id=sequence_cfg["subject_id"],
        least_avail_markers=sequence_cfg["least_avail_markers"],
        optimize_fingers=sequence_cfg["optimize_fingers"],
        end_frame=120 if mode == "smoke" else -1,
    )


def build_capture_overrides(
    c3d: Path,
    marker_layout: Path,
    output_dir: Path,
    *,
    model_dir: Path = MODEL_DIR,
    source_units: str = "m",
    source_up: str = "z",
    subject_id: int = 0,
    sample_step: int = 4,
    end_frame: int = -1,
    optimize_fingers: bool = False,
    least_avail_markers: float = 0.95,
) -> dict:
    """Apply the legacy Stage I/II settings to an explicitly supplied capture."""
    if source_units not in {"m", "mm", "cm"} or source_up not in {"y", "z"}:
        raise ValueError("Specify supported source units and an explicit Y/Z up axis")
    if sample_step < 1 or not 0 < least_avail_markers <= 1:
        raise ValueError("Invalid sampling step or marker availability threshold")
    markers = load_layout_markers(Path(marker_layout))
    return {
        "mocap.fname": str(Path(c3d).resolve()),
        "mocap.unit": source_units,
        # The source C3D is Z-up; SMPL-X/MoSh++ expects Y-up.
        "mocap.rotate": [-90, 0, 0] if source_up == "z" else [0, 0, 0],
        "mocap.subject_id": subject_id,
        # This excludes unlabeled/Active tracks and is derived from the layout.
        "mocap.only_markers": markers,
        "mocap.start_fidx": 0,
        # Smoke: first 120 source frames sampled at 30 Hz -> 30 fitted frames.
        # Full: all source frames sampled at 30 Hz.
        "mocap.end_fidx": end_frame,
        "mocap.ds_rate": sample_step,
        "surface_model.type": "smplx",
        "surface_model.gender": "neutral",
        "surface_model.fname": str(
            Path(model_dir).resolve() / "SMPLX_NEUTRAL_moshpp.pkl"
        ),
        "surface_model.dmpl_fname": None,
        "surface_model.num_betas": 16,
        "surface_model.dof_per_hand": 24,
        "surface_model.use_hands_mean": True,
        "moshpp.pose_body_prior_fname": str(
            Path(model_dir).resolve() / "moshpp_support/pose_body_prior.pkl"
        ),
        "moshpp.pose_hand_prior_fname": str(
            Path(model_dir).resolve() / "moshpp_support/pose_hand_prior.npz"
        ),
        "moshpp.head_marker_corr_fname": None,
        "moshpp.perseq_mosh_stagei": True,
        "moshpp.optimize_fingers": optimize_fingers,
        "moshpp.optimize_face": False,
        "moshpp.optimize_toes": False,
        "moshpp.optimize_betas": True,
        "moshpp.optimize_dynamics": False,
        "moshpp.verbosity": 1,
        "moshpp.stagei_frame_picker.least_avail_markers": least_avail_markers,
        "dirs.support_base_dir": str(ROOT / "src/moshpp/support_data"),
        "dirs.work_base_dir": str(Path(output_dir).resolve()),
        "dirs.marker_layout.fname": str(Path(marker_layout).resolve()),
        "runtime.stagei_only": False,
    }
