#!/usr/bin/env python
"""Fit the supported Zekun C3D sequences to the neutral SMPL-X model."""

import argparse
import json
from pathlib import Path

import numpy as np

from moshpp.mosh_head import MoSh, run_moshpp_once


ROOT = Path("/media/fiona/LinuxData/mocap_retarget")
MODEL_DIR = ROOT / "mocap_data/data/smplx"
HEAD_MARKERS = ("HeadTop", "HeadFront", "HeadSide")

SEQUENCES = {
    "upper": {
        "c3d": ROOT / "mocap_data/data/1112/c3d/Zekun_upper_1.c3d",
        "marker_layout": ROOT
        / "mocap_data/data/1112/c3d/marker_layout/Zekun_upper_25/marker_layout.json",
        "subject_id": 0,
        "least_avail_markers": 1.0,
        "optimize_fingers": False,
        "output_suffix": "",
    },
    "righthand": {
        "c3d": ROOT
        / "mocap_data/data/1112/c3d/Zekun_upper_righthand_1.c3d",
        "marker_layout": ROOT
        / "mocap_data/data/1112/c3d/marker_layout/Zekun_upper_righthand_32/marker_layout.json",
        # Body and hand markers live in separate Motive namespaces. -1 keeps both.
        "subject_id": -1,
        # Three wrist/hand labels occur in both namespaces, so the C3D contains
        # 35 tracks for 32 unique labels. No frame has all 35 tracks visible.
        "least_avail_markers": 0.95,
        "optimize_fingers": True,
        "output_suffix": "_fingers",
    },
    "righthand-added": {
        "c3d": ROOT
        / "mocap_data/data/1112/c3d/Zekun_upper_righthand_added_1.c3d",
        "marker_layout": ROOT
        / "mocap_data/data/1112/c3d/marker_layout/Zekun_upper_righthand_added_32/marker_layout.json",
        "subject_id": 0,
        "least_avail_markers": 1.0,
        "optimize_fingers": True,
        "output_suffix": "_fingers",
        # This take begins with a measured upright, stationary calibration
        # pose. Put it first, then retain diverse frames for shape/marker fit.
        "head_calibration_frame": 0,
        "stagei_frame_ids": (0, 1052, 1783, 75, 27, 856, 110, 908, 241, 173, 31, 1158),
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
        raise ValueError("Marker layout contains no assigned markers: {}".format(marker_layout))
    if len(markers) != len(set(markers)):
        raise ValueError("Marker layout contains duplicate labels: {}".format(marker_layout))
    return markers


def build_overrides(mode, sequence):
    is_smoke = mode == "smoke"
    sequence_cfg = SEQUENCES[sequence]
    c3d = sequence_cfg["c3d"]
    marker_layout = sequence_cfg["marker_layout"]
    markers = load_layout_markers(marker_layout)
    head_calibration_frame = sequence_cfg.get("head_calibration_frame")
    stagei_frame_ids = sequence_cfg.get("stagei_frame_ids")
    if head_calibration_frame is None:
        output_name = "{}{}_{}".format(
            c3d.stem, sequence_cfg["output_suffix"], mode
        )
    else:
        if not stagei_frame_ids or head_calibration_frame not in stagei_frame_ids:
            raise ValueError("The head calibration frame must be selected for Stage I")
        output_name = "{}{}_head_calibrated_{}".format(
            c3d.stem, sequence_cfg["output_suffix"], mode
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
        "dirs.support_base_dir": str(ROOT / "moshpp/support_data"),
        "dirs.work_base_dir": str(ROOT / "moshpp_outputs" / output_name),
        "dirs.marker_layout.fname": str(marker_layout),
        "runtime.stagei_only": False,
    }

    if head_calibration_frame is not None:
        stagei_frame_names = [f"{c3d}_{frame_id}" for frame_id in stagei_frame_ids]
        # Keep the three marker locations optimizable. Only the known-upright
        # calibration frame anchors the SMPL-X neck/head rotations, resolving
        # the constant-rotation ambiguity without freezing marker placement.
        overrides.update(
            {
                "moshpp.head_marker_labels": list(HEAD_MARKERS),
                "moshpp.head_marker_max_offset": None,
                "moshpp.head_pose_anchor_frame_indices": [
                    stagei_frame_ids.index(head_calibration_frame)
                ],
                "moshpp.stagei_frame_picker.type": "manual",
                "moshpp.stagei_frame_picker.num_frames": len(stagei_frame_names),
                "moshpp.stagei_frame_picker.stagei_mocap_fnames": stagei_frame_names,
            }
        )

    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        choices=tuple(SEQUENCES),
        default="upper",
        help="Input C3D/layout profile (default: upper).",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--stagei-only",
        action="store_true",
        help="Stop after Stage I so its calibration can be inspected.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Resolve paths and configuration without starting optimization.",
    )
    args = parser.parse_args()

    sequence_cfg = SEQUENCES[args.sequence]
    c3d = sequence_cfg["c3d"]
    marker_layout = sequence_cfg["marker_layout"]
    for required_path in (
        c3d,
        marker_layout,
        MODEL_DIR / "SMPLX_NEUTRAL_moshpp.pkl",
        MODEL_DIR / "moshpp_support/pose_body_prior.pkl",
        MODEL_DIR / "moshpp_support/pose_hand_prior.npz",
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    overrides = build_overrides(args.mode, args.sequence)
    overrides["runtime.stagei_only"] = args.stagei_only
    resolved = MoSh.prepare_cfg(**overrides)
    stage_i = Path(str(resolved.dirs.stagei_fname))
    stage_ii = Path(str(resolved.dirs.stageii_fname))
    output_npz = stage_ii.with_suffix(".npz")

    print("Sequence:", args.sequence)
    print("Mode:", args.mode)
    print("C3D:", c3d)
    print("Marker layout:", marker_layout)
    print("Marker labels:", len(overrides["mocap.only_markers"]))
    print("Optimize fingers:", overrides["moshpp.optimize_fingers"])
    print(
        "Selected subject:",
        resolved.mocap.subject_name
        if resolved.mocap.subject_name is not None
        else "all namespaces",
    )
    print(
        "Stage I minimum raw-track availability:",
        overrides["moshpp.stagei_frame_picker.least_avail_markers"],
    )
    print("Source frame rate: 120 Hz")
    print("Output frame rate: 30 Hz")
    print("Stage I:", stage_i)
    print("Stage II:", stage_ii)
    print("AMASS-style NPZ:", output_npz)

    if args.check_only:
        print("Configuration check OK; optimization was not started.")
        return

    run_moshpp_once(overrides)

    if args.stagei_only:
        if not stage_i.exists():
            raise RuntimeError("Stage I did not produce the expected file: {}".format(stage_i))
        print("Stage I completed successfully; Stage II was not started.")
        print("Saved:", stage_i)
        return

    if not stage_ii.exists():
        raise RuntimeError("Stage II did not produce the expected file: {}".format(stage_ii))

    result = MoSh.load_as_amass_npz(
        stage_ii,
        stageii_npz_fname=output_npz,
        include_markers=True,
        include_extra_details=True,
    )

    # MoSh++ retains the source C3D frame rate after temporal downsampling.
    # Store metadata for the actual parameter sequence exported here.
    output_frame_rate = float(result["mocap_frame_rate"]) / overrides["mocap.ds_rate"]
    result["mocap_frame_rate"] = output_frame_rate
    result["mocap_time_length"] = len(result["trans"]) / output_frame_rate
    np.savez(output_npz, **result)

    print("MoSh++ completed successfully.")
    print("frame rate:", result["mocap_frame_rate"])
    print("trans:", np.asarray(result["trans"]).shape)
    print("poses:", np.asarray(result["poses"]).shape)
    print("betas:", np.asarray(result["betas"]).shape)
    print("Saved:", output_npz)


if __name__ == "__main__":
    main()
