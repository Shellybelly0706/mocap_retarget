"""Reusable smplx head markers helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEAD_LABELS = ("HeadTop", "HeadFront", "HeadSide")


COLORS = {
    "HeadTop": "#e69f00",
    "HeadFront": "#0072b2",
    "HeadSide": "#009e73",
}


def observed_and_simulated(
    archive: np.lib.npyio.NpzFile, label: str
) -> tuple[np.ndarray, np.ndarray]:
    observed: list[np.ndarray] = []
    simulated: list[np.ndarray] = []
    for labels, obs, sim in zip(
        archive["labels_obs"], archive["markers_obs"], archive["markers_sim"]
    ):
        labels = np.asarray(labels).astype(str)
        index = np.flatnonzero(labels == label)
        if index.size:
            observed.append(np.asarray(obs, dtype=np.float64)[index[0]])
            simulated.append(np.asarray(sim, dtype=np.float64)[index[0]])
    return np.asarray(observed), np.asarray(simulated)


def diagnose_head_markers(
    result_path: Path, model_path: Path, output_path: Path
) -> None:
    """Render head-marker diagnostics for explicit input and output paths."""
    import matplotlib.pyplot as plt
    import torch
    from human_body_prior.body_model.body_model import BodyModel

    result_path = Path(result_path).resolve()
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()

    with np.load(result_path, allow_pickle=True) as archive:
        betas = np.asarray(archive["betas"], dtype=np.float32)
        latent = np.asarray(archive["markers_latent"], dtype=np.float64)
        latent_labels = np.asarray(archive["latent_labels"]).astype(str)
        marker_vids = archive["marker_meta"].item()["marker_vids"]
        head_data = {
            label: observed_and_simulated(archive, label) for label in HEAD_LABELS
        }

    model = BodyModel(
        bm_fname=str(model_path), num_betas=len(betas), model_type="smplx"
    )
    with torch.no_grad():
        vertices = model(betas=torch.from_numpy(betas[None])).v[0].cpu().numpy()

    expected: dict[str, np.ndarray] = {}
    inferred: dict[str, np.ndarray] = {}
    displacement_mm: dict[str, float] = {}
    residual_rms_mm: dict[str, float] = {}
    for label in HEAD_LABELS:
        expected[label] = vertices[int(marker_vids[label])]
        latent_index = np.flatnonzero(latent_labels == label)
        if latent_index.size != 1:
            raise ValueError(f"Expected one latent marker named {label!r}")
        inferred[label] = latent[latent_index[0]]
        displacement_mm[label] = 1000.0 * float(
            np.linalg.norm(inferred[label] - expected[label])
        )
        obs, sim = head_data[label]
        residual_rms_mm[label] = 1000.0 * float(
            np.sqrt(np.mean(np.sum((obs - sim) ** 2, axis=1)))
        )

    obs_triangle = np.stack([head_data[label][0] for label in HEAD_LABELS], axis=1)
    first, second, third = (obs_triangle[:, i] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(second - first, third - first), axis=1)
    max_side = np.maximum.reduce(
        (
            np.linalg.norm(first - second, axis=1),
            np.linalg.norm(first - third, axis=1),
            np.linalg.norm(second - third, axis=1),
        )
    )
    minimum_altitude_mm = 1000.0 * float(np.median(2.0 * area / max_side))

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    top_ax = fig.add_subplot(grid[0, 0])
    front_ax = fig.add_subplot(grid[0, 1])
    bar_ax = fig.add_subplot(grid[0, 2])
    text_ax = fig.add_subplot(grid[1, :])

    head_vertices = vertices[vertices[:, 1] > 0.19]
    projections = (
        (
            top_ax,
            0,
            2,
            "Top view of canonical head",
            "left/right X (m)",
            "front/back Z (m)",
        ),
        (
            front_ax,
            0,
            1,
            "Front view of canonical head",
            "left/right X (m)",
            "height Y (m)",
        ),
    )
    for axis, horizontal, vertical, title, xlabel, ylabel in projections:
        axis.scatter(
            head_vertices[:, horizontal],
            head_vertices[:, vertical],
            s=2,
            color="#d0d0d0",
            alpha=0.35,
            rasterized=True,
        )
        for label in HEAD_LABELS:
            start = expected[label]
            end = inferred[label]
            color = COLORS[label]
            axis.annotate(
                "",
                xy=(end[horizontal], end[vertical]),
                xytext=(start[horizontal], start[vertical]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
            )
            axis.scatter(
                start[horizontal],
                start[vertical],
                s=85,
                facecolors="none",
                edgecolors=color,
                linewidths=2,
                marker="o",
            )
            axis.scatter(
                end[horizontal],
                end[vertical],
                s=85,
                color=color,
                marker="x",
                linewidths=2.5,
            )
            axis.text(
                end[horizontal] + 0.004,
                end[vertical] + 0.004,
                label,
                fontsize=9,
                color=color,
            )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    labels = list(HEAD_LABELS)
    positions = np.arange(len(labels))
    bars = bar_ax.bar(
        positions,
        [displacement_mm[label] for label in labels],
        color=[COLORS[label] for label in labels],
        alpha=0.85,
    )
    bar_ax.bar_label(bars, fmt="%.1f mm", padding=3)
    bar_ax.set_xticks(positions, labels, rotation=20)
    bar_ax.set_ylabel("Stage-I shift from assigned vertex (mm)")
    bar_ax.set_title("Anatomical correspondence shift")
    bar_ax.set_ylim(0, max(displacement_mm.values()) * 1.2)
    bar_ax.grid(axis="y", alpha=0.2)

    text_ax.axis("off")
    rms_text = ", ".join(
        f"{label} {residual_rms_mm[label]:.2f} mm" for label in HEAD_LABELS
    )
    shift_text = ", ".join(
        f"{label} {displacement_mm[label]:.1f} mm" for label in HEAD_LABELS
    )
    explanation = (
        "Finding: the point count is adequate, but the marker-to-anatomy calibration is not.\n\n"
        f"All three head markers are available in {len(obs_triangle)} frames. "
        "Their observed triangle has a median "
        f"minimum altitude of {minimum_altitude_mm:.1f} mm, so it is far from "
        "collinear and can determine a rigid "
        "orientation. Three non-collinear points are mathematically sufficient.\n\n"
        f"The fit residuals are small ({rms_text}), yet Stage I relocates the "
        "virtual markers far from the manually "
        f"assigned anatomical vertices ({shift_text}). Open circles show the "
        "expected layout; crosses show the "
        "Stage-I inferred locations. In particular, HeadFront moves toward the "
        "side of the skull and HeadSide moves "
        "toward the back. This consistent correspondence/calibration mismatch "
        "creates a constant head-orientation "
        "bias even though the optimizer can match the measured dots."
    )
    text_ax.text(
        0.015,
        0.96,
        explanation,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.45,
        wrap=True,
    )
    fig.suptitle(
        "SMPL-X head-marker diagnosis: expected anatomical positions vs MoSh++ Stage-I locations",
        fontsize=16,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Observed head frames: {len(obs_triangle)}")
    print(f"Median observed triangle minimum altitude: {minimum_altitude_mm:.3f} mm")
    for label in HEAD_LABELS:
        print(
            f"{label}: Stage-I shift={displacement_mm[label]:.3f} mm, "
            f"fit RMS={residual_rms_mm[label]:.3f} mm"
        )
