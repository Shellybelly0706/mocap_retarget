"""Reusable smplx fitting helpers."""

from __future__ import annotations

import json

from utils.paths import REPOSITORY_ROOT as _REPOSITORY_ROOT

ROOT = _REPOSITORY_ROOT


MODEL_DIR = ROOT / "data/smplx"


SEQUENCES = {
    "upper": {
        "c3d": ROOT / "data/1112/c3d/Zekun_upper_1.c3d",
        "marker_layout": ROOT
        / "data/1112/c3d/marker_layout/Zekun_upper_25/marker_layout.json",
        "subject_id": 0,
        "least_avail_markers": 1.0,
        "optimize_fingers": False,
        "output_suffix": "",
        "output_tag": "layout_fixed_v2",
    },
    "righthand": {
        "c3d": ROOT / "data/1112/c3d/Zekun_upper_righthand_1.c3d",
        "marker_layout": ROOT
        / "data/1112/c3d/marker_layout/Zekun_upper_righthand_32/marker_layout.json",
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
        "marker_layout": ROOT
        / "data/1112/c3d/marker_layout/Zekun_upper_righthand_added_32/marker_layout.json",
        "subject_id": 0,
        "least_avail_markers": 1.0,
        "optimize_fingers": True,
        "output_suffix": "_fingers",
        "output_tag": "layout_fixed_v2",
    },
}


def load_layout_markers(marker_layout):
    """Return assigned marker labels from a MoSh++ marker-layout JSON file."""
    with marker_layout.open("r", encoding="utf-8") as handle:
        layout = json.load(handle)

    markers = []
    unassigned = []
    for marker_set in layout.get("markersets", []):
        for label, vertex_id in marker_set.get("indices", {}).items():
            if vertex_id is None:
                unassigned.append(label)
            else:
                markers.append(label)

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
    is_smoke = mode == "smoke"
    sequence_cfg = SEQUENCES[sequence]
    c3d = sequence_cfg["c3d"]
    marker_layout = sequence_cfg["marker_layout"]
    markers = load_layout_markers(marker_layout)
    output_name = "{}{}_{}_{}".format(
        c3d.stem, sequence_cfg["output_suffix"], sequence_cfg["output_tag"], mode
    )

    overrides = {
        "mocap.fname": str(c3d),
        "mocap.unit": "m",
        # The source C3D is Z-up; SMPL-X/MoSh++ expects Y-up.
        "mocap.rotate": [-90, 0, 0],
        "mocap.subject_id": sequence_cfg["subject_id"],
        # This excludes unlabeled/Active tracks and is derived from the layout.
        "mocap.only_markers": markers,
        "mocap.start_fidx": 0,
        # Smoke: first 120 source frames sampled at 30 Hz -> 30 fitted frames.
        # Full: all source frames sampled at 30 Hz.
        "mocap.end_fidx": 120 if is_smoke else -1,
        "mocap.ds_rate": 4,
        "surface_model.type": "smplx",
        "surface_model.gender": "neutral",
        "surface_model.fname": str(MODEL_DIR / "SMPLX_NEUTRAL_moshpp.pkl"),
        "surface_model.dmpl_fname": None,
        "surface_model.num_betas": 16,
        "surface_model.dof_per_hand": 24,
        "surface_model.use_hands_mean": True,
        "moshpp.pose_body_prior_fname": str(
            MODEL_DIR / "moshpp_support/pose_body_prior.pkl"
        ),
        "moshpp.pose_hand_prior_fname": str(
            MODEL_DIR / "moshpp_support/pose_hand_prior.npz"
        ),
        "moshpp.head_marker_corr_fname": None,
        "moshpp.perseq_mosh_stagei": True,
        "moshpp.optimize_fingers": sequence_cfg["optimize_fingers"],
        "moshpp.optimize_face": False,
        "moshpp.optimize_toes": False,
        "moshpp.optimize_betas": True,
        "moshpp.optimize_dynamics": False,
        "moshpp.verbosity": 1,
        "moshpp.stagei_frame_picker.least_avail_markers": sequence_cfg[
            "least_avail_markers"
        ],
        "dirs.support_base_dir": str(ROOT / "src/moshpp/support_data"),
        "dirs.work_base_dir": str(ROOT / "moshpp_outputs" / output_name),
        "dirs.marker_layout.fname": str(marker_layout),
        "runtime.stagei_only": False,
    }

    return overrides
