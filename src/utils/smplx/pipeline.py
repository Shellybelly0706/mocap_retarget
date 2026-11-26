"""Run the legacy MoSh++ Stage I/II workflow without mixing it into the viewer.

Preflight is data-only and works in mocap_data. Actual optimization requires
the shared mocap_data environment and a complete marker-to-surface layout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from utils.markers.layout_io import read_json, sha256_file
from utils.optitrack.c3d import UNIT_TO_METERS, load_c3d
from utils.paths import MOSHPP_DIR, SMPLX_ASSETS_DIR
from utils.smplx.fitting import build_capture_overrides, load_layout_indices

RUNTIME_MODULES = (
    "chumpy",
    "torch",
    "human_body_prior",
    "body_visualizer",
    "ezc3d",
    "omegaconf",
    "psbody.mesh",
    "psbody.smpl.fast_derivatives.smplcpp_chumpy",
    "moshpp.scan2mesh.mesh_distance.sample2meshdist",
)


def runtime_status() -> dict:
    modules = {}
    for name in RUNTIME_MODULES:
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            modules[name] = False
    compatible = (3, 7) <= sys.version_info[:2] <= (3, 10) and tuple(
        map(int, np.__version__.split(".")[:2])
    ) < (1, 24)
    return {
        "python": sys.version.split()[0],
        "python_prefix": sys.prefix,
        "numpy": np.__version__,
        "compatible_python_numpy": compatible,
        "modules": modules,
        "ready": compatible and all(modules.values()),
    }


def preflight(
    c3d: Path,
    layout: Path,
    *,
    model_dir: Path = SMPLX_ASSETS_DIR,
    subject_prefix: str | None = None,
    mode: str = "smoke",
    sample_step: int = 4,
    least_avail_markers: float = 0.95,
) -> dict:
    """Inspect real observations, layout coverage, model files and runtime."""
    if (
        mode not in {"smoke", "full"}
        or sample_step < 1
        or not 0 < least_avail_markers <= 1
    ):
        raise ValueError("Invalid fitting mode, sample step or marker availability")
    c3d, layout, model_dir = map(Path, (c3d, layout, model_dir))
    motion = load_c3d(c3d)
    blockers = []
    required = [
        model_dir / name
        for name in (
            "SMPLX_NEUTRAL_moshpp.pkl",
            "moshpp_support/pose_body_prior.pkl",
            "moshpp_support/pose_hand_prior.npz",
        )
    ]
    for path in required:
        if not path.is_file():
            blockers.append(f"Missing fitting asset: {path}")
        else:
            with path.open("rb") as stream:
                if stream.read(80).startswith(
                    b"version https://git-lfs.github.com/spec/"
                ):
                    blockers.append(f"LFS pointer needs download: {path}")
    indices = load_layout_indices(layout) if layout.is_file() else {}
    if not indices:
        blockers.append(f"Missing or empty marker layout: {layout}")
    unassigned = [name for name, vertex in indices.items() if vertex is None]
    invalid_vertices = [
        name
        for name, vertex in indices.items()
        if vertex is not None and not 0 <= vertex < 10475
    ]
    if unassigned:
        blockers.append(f"{len(unassigned)} marker-to-surface assignments are unset")
    if invalid_vertices:
        blockers.append(
            "Out-of-range SMPL-X marker vertex IDs: " + ", ".join(invalid_vertices)
        )
    prefix = subject_prefix.rstrip(":") + ":" if subject_prefix else None
    lookup = {}
    for index, label in enumerate(motion.labels):
        if prefix and not label.startswith(prefix):
            continue
        lookup.setdefault(label.rsplit(":", 1)[-1], []).append(index)
    missing = [name for name in indices if name not in lookup]
    ambiguous = [name for name in indices if len(lookup.get(name, [])) > 1]
    if missing:
        blockers.append(
            "Layout markers absent from selected capture namespace: "
            + ", ".join(missing)
        )
    if ambiguous:
        blockers.append(
            "Ambiguous duplicate marker names; specify subject prefix: "
            + ", ".join(ambiguous)
        )
    assigned = [
        name
        for name, vertex in indices.items()
        if vertex is not None
        and name in lookup
        and len(lookup[name]) == 1
        and name not in invalid_vertices
    ]
    selected_columns = [lookup[name][0] for name in assigned]
    coverage = {
        name: float(motion.valid[:, lookup[name][0]].mean())
        for name in indices
        if name in lookup and len(lookup[name]) == 1
    }
    eligible = 0
    if selected_columns:
        eligible = int(
            (
                motion.valid[:, selected_columns].mean(axis=1) >= least_avail_markers
            ).sum()
        )
        if eligible < 12:
            blockers.append(
                f"Only {eligible} frames meet the Stage I marker availability threshold; 12 are needed"
            )
    runtime = runtime_status()
    if not runtime["compatible_python_numpy"]:
        blockers.append(
            "Legacy fitting needs Python <=3.10 and NumPy <1.24; update mocap_data from environment.yml"
        )
    missing_runtime = [
        name for name, available in runtime["modules"].items() if not available
    ]
    if missing_runtime:
        blockers.append("Missing legacy runtime modules: " + ", ".join(missing_runtime))
    end = min(len(motion.positions), 120) if mode == "smoke" else len(motion.positions)
    rows = np.arange(0, end, sample_step, dtype=int)
    return {
        "method": "legacy MoSh++ marker-based Stage I shape/calibration, Stage II motion",
        "source_c3d": str(c3d.resolve()),
        "source_sha256": sha256_file(c3d),
        "source_units": motion.source_units,
        "source_frame_rate": motion.frame_rate,
        "source_frame_count": len(motion.positions),
        "source_up_axis": "declared separately in fitting arguments",
        "subject_prefix": subject_prefix,
        "marker_layout": str(layout.resolve()),
        "marker_layout_sha256": sha256_file(layout) if layout.is_file() else None,
        "layout_marker_count": len(indices),
        "assigned_markers": assigned,
        "unassigned_markers": unassigned,
        "missing_markers": missing,
        "excluded_capture_markers": sorted(set(lookup).difference(indices)),
        "marker_valid_fraction": coverage,
        "stagei_eligible_frames": eligible,
        "stagei_least_avail_markers": least_avail_markers,
        "mode": mode,
        "sample_step": sample_step,
        "expected_fitted_frames": len(rows),
        "source_sample_rows": rows.tolist(),
        "source_frame_numbers": motion.frame_numbers[rows].tolist(),
        "source_relative_time_s": (
            (motion.frame_numbers[rows] - motion.frame_numbers[0]) / motion.frame_rate
        ).tolist(),
        "output_frame_rate": motion.frame_rate / sample_step,
        "runtime": runtime,
        "blockers": blockers,
        "ready": not blockers,
        "fit_started": False,
        "clock_policy": "C3D-relative timing only; no global timestamp or IMU clock adjustment",
    }


def run_legacy_fit(
    c3d: Path,
    layout: Path,
    output: Path,
    *,
    model_dir: Path = SMPLX_ASSETS_DIR,
    subject_prefix: str | None = None,
    subject_id: int = 0,
    source_up: str = "z",
    mode: str = "smoke",
    sample_step: int = 4,
    optimize_fingers: bool = False,
    least_avail_markers: float = 0.95,
    stagei_only: bool = False,
) -> dict:
    """Run the same Stage I/II solver and AMASS export as the removed launcher."""
    report = preflight(
        c3d,
        layout,
        model_dir=model_dir,
        subject_prefix=subject_prefix,
        mode=mode,
        sample_step=sample_step,
        least_avail_markers=least_avail_markers,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report["source_up_axis"] = source_up
    report_path = output / "fitting_report.json"
    (output / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["ready"]:
        raise RuntimeError("MoSh++ preflight failed: " + "; ".join(report["blockers"]))
    scale = UNIT_TO_METERS[report["source_units"].lower()]
    units = {1.0: "m", 0.001: "mm", 0.01: "cm"}[scale]
    overrides = build_capture_overrides(
        c3d,
        layout,
        output,
        model_dir=model_dir,
        source_units=units,
        source_up=source_up,
        subject_id=subject_id,
        sample_step=sample_step,
        end_frame=min(120, report["source_frame_count"]) if mode == "smoke" else -1,
        optimize_fingers=optimize_fingers,
        least_avail_markers=least_avail_markers,
    )
    overrides["runtime.stagei_only"] = stagei_only
    identity = {
        "capture_sha256": report["source_sha256"],
        "layout_sha256": report["marker_layout_sha256"],
        "model_dir": str(Path(model_dir).resolve()),
        "subject_id": subject_id,
        "source_up": source_up,
        "mode": mode,
        "sample_step": sample_step,
        "optimize_fingers": optimize_fingers,
        "least_avail_markers": least_avail_markers,
    }
    if report_path.exists() and read_json(report_path).get("job_identity") != identity:
        raise ValueError(
            "Output belongs to a different fitting job; choose a new output directory"
        )
    report["job_identity"] = identity
    # Keep the user-supplied/reference layout intact; stage I may optimize a copy.
    fitting_layout = output / "fitting_marker_layout.json"
    if fitting_layout.resolve() == Path(layout).resolve():
        raise ValueError("Input layout must differ from the optimized working copy")
    if fitting_layout.exists() and not report_path.exists():
        raise ValueError(
            "Output contains a layout without job provenance; choose a new output directory"
        )
    if not fitting_layout.exists():
        fitting_layout.write_bytes(Path(layout).read_bytes())
    overrides["dirs.marker_layout.fname"] = str(fitting_layout.resolve())
    from moshpp.mosh_head import MoSh, run_moshpp_once

    resolved = MoSh.prepare_cfg(**overrides)
    report["fit_started"] = True
    report["overrides"] = overrides
    report["solver_source"] = str(MOSHPP_DIR)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    run_moshpp_once(overrides)
    stage_i, stage_ii = Path(str(resolved.dirs.stagei_fname)), Path(
        str(resolved.dirs.stageii_fname)
    )
    report["stagei_result"] = str(stage_i)
    if not stage_i.is_file():
        raise RuntimeError("MoSh++ did not produce the expected Stage I result")
    if not stagei_only:
        if not stage_ii.is_file():
            raise RuntimeError("MoSh++ did not produce the expected Stage II result")
        result = MoSh.load_as_amass_npz(
            stage_ii, include_markers=True, include_extra_details=True
        )
        if len(result["trans"]) != report["expected_fitted_frames"]:
            raise RuntimeError(
                "Fitted frame count differs from the explicit source-sample mapping"
            )
        result["mocap_frame_rate"] = report["output_frame_rate"]
        result["mocap_time_length"] = len(result["trans"]) / report["output_frame_rate"]
        result["source_sample_rows"] = np.asarray(
            report["source_sample_rows"], dtype=np.int64
        )
        result["source_frame_numbers"] = np.asarray(
            report["source_frame_numbers"], dtype=np.int64
        )
        result["source_relative_time_s"] = np.asarray(
            report["source_relative_time_s"], dtype=float
        )
        result["source_frame_rate"] = report["source_frame_rate"]
        target = stage_ii.with_suffix(".npz")
        np.savez_compressed(target, **result)
        report["stageii_result"] = str(stage_ii)
        report["smplx_parameters"] = str(target)
    report["fit_completed"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
