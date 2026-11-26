"""Reusable markers layout helpers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.markers.layout_io import (
    atomic_json_write,
    backup_once,
    exact_official_matches,
    read_json,
)
from utils.markers.layout_models import (
    DEFAULT_DISTANCE_FROM_SKIN_M,
    MeshData,
    ValidationResult,
)


def make_initial_layout(
    labels: Sequence[str],
    source_path: Path,
    moshpp_root: Path,
    auto_match: bool,
    reference_layout_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    matches: dict[str, int] = {}
    sources: dict[str, dict[str, Any]] = {}
    marker_file = moshpp_root / "src/moshpp/marker_layout/marker_vids.py"
    alias_file = moshpp_root / "src/moshpp/marker_layout/labels_map.py"
    if auto_match:
        matches, sources, marker_file, alias_file = exact_official_matches(
            labels, moshpp_root
        )
    reference_metadata: Mapping[str, Any] = {}
    if reference_layout_path is not None:
        reference_layout = read_json(reference_layout_path)
        if reference_layout.get("surface_model_type") != "smplx":
            raise ValueError(
                "Reference layout surface_model_type must be exactly 'smplx'"
            )
        reference_indices = get_indices(reference_layout)
        reference_sources = reference_layout.get("marker_sources", {})
        reference_metadata_value = reference_layout.get("layout_metadata", {})
        if isinstance(reference_metadata_value, Mapping):
            reference_metadata = reference_metadata_value
        for label in labels:
            vertex = reference_indices.get(label)
            if vertex is None:
                continue
            matches[label] = int(vertex)
            source: dict[str, Any] = {
                "method": "reference_layout_exact_label",
                "reference_layout": str(reference_layout_path.resolve()),
                "reference_label": label,
            }
            if isinstance(reference_sources, Mapping) and label in reference_sources:
                source["reference_source"] = reference_sources[label]
            sources[label] = source
    indices = {label: matches.get(label) for label in labels}
    layout: dict[str, Any] = {
        "surface_model_type": "smplx",
        "markersets": [
            {
                "type": "body",
                "distance_from_skin": DEFAULT_DISTANCE_FROM_SKIN_M,
                "indices": indices,
            }
        ],
        "marker_sources": sources,
        "layout_metadata": {
            "source_path": str(source_path.resolve()),
            "source_format": source_path.suffix.lower().lstrip("."),
            "official_marker_database": str(marker_file.resolve()),
            "official_alias_database": str(alias_file.resolve()),
            "reference_layout": (
                str(reference_layout_path.resolve())
                if reference_layout_path is not None
                else None
            ),
            "model_path": reference_metadata.get("model_path"),
            "model_sha256": reference_metadata.get("model_sha256"),
            "topology_vertex_count": reference_metadata.get("topology_vertex_count"),
            "policy": (
                "Only exact official labels/aliases and exact labels from the "
                "explicit reference layout are auto-filled; all other vertices "
                "require manual surface clicks and may remain null."
            ),
        },
    }
    return layout, matches


def initialize_layout(
    layout_path: Path,
    labels: Sequence[str],
    source_path: Path,
    moshpp_root: Path,
    auto_match: bool = True,
    reference_layout_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, int], Path | None]:
    """Initialize only when absent or non-JSON; preserve any valid JSON layout."""
    if layout_path.exists():
        try:
            existing = read_json(layout_path)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            backup = backup_once(layout_path)
            layout, matches = make_initial_layout(
                labels,
                source_path,
                moshpp_root,
                auto_match,
                reference_layout_path,
            )
            atomic_json_write(layout_path, layout)
            return layout, matches, backup
        return existing, {}, None
    layout, matches = make_initial_layout(
        labels,
        source_path,
        moshpp_root,
        auto_match,
        reference_layout_path,
    )
    atomic_json_write(layout_path, layout)
    return layout, matches, None


def get_indices(layout: Mapping[str, Any]) -> dict[str, int | None]:
    markersets = layout.get("markersets")
    if not isinstance(markersets, list) or not markersets:
        raise ValueError("Layout must contain a non-empty markersets list")
    result: dict[str, int | None] = {}
    for markerset in markersets:
        if not isinstance(markerset, dict):
            raise ValueError("Every markerset must be a JSON object")
        if not isinstance(markerset.get("type"), str) or not markerset["type"]:
            raise ValueError("Every markerset needs a non-empty type")
        distance = markerset.get("distance_from_skin", DEFAULT_DISTANCE_FROM_SKIN_M)
        if not isinstance(distance, (int, float)) or not math.isfinite(float(distance)):
            raise ValueError("distance_from_skin must be finite")
        if not math.isclose(float(distance), 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "distance_from_skin must be 0.0: this editor only permits "
                "markers exactly on the SMPL-X surface"
            )
        indices = markerset.get("indices")
        if not isinstance(indices, dict):
            raise ValueError("Every markerset needs an indices object")
        for label, vertex in indices.items():
            if label in result:
                raise ValueError(f"Marker label appears more than once: {label}")
            if vertex is not None and (
                isinstance(vertex, bool) or not isinstance(vertex, int)
            ):
                raise ValueError(f"Vertex ID for {label!r} must be an integer or null")
            result[str(label)] = vertex
    return result


def set_index(layout: dict[str, Any], label: str, vertex: int | None) -> None:
    for markerset in layout["markersets"]:
        if label in markerset["indices"]:
            markerset["indices"][label] = vertex
            return
    raise KeyError(label)


def validate_layout(
    layout: Mapping[str, Any],
    expected_labels: Sequence[str],
    mesh: MeshData | None,
    require_label_order: bool = True,
) -> ValidationResult:
    if layout.get("surface_model_type") != "smplx":
        raise ValueError("surface_model_type must be exactly 'smplx'")
    indices = get_indices(layout)
    labels = tuple(indices)
    expected = tuple(expected_labels)
    labels_match = (
        labels == expected if require_label_order else set(labels) == set(expected)
    )
    if not labels_match:
        missing = [label for label in expected if label not in indices]
        extra = [label for label in labels if label not in set(expected)]
        order_only = not missing and not extra
        raise ValueError(
            "Layout labels do not match source labels; "
            f"missing={missing}, extra={extra}, order_only_mismatch={order_only}"
        )
    assigned = tuple(label for label, vertex in indices.items() if vertex is not None)
    unassigned = tuple(label for label, vertex in indices.items() if vertex is None)
    vertex_count = len(mesh.vertices) if mesh is not None else None
    if assigned and mesh is None:
        raise ValueError(
            "A model is required to verify non-null vertex IDs against the exact topology"
        )
    if mesh is not None:
        invalid = {
            label: vertex
            for label, vertex in indices.items()
            if vertex is not None and not 0 <= int(vertex) < vertex_count
        }
        if invalid:
            raise ValueError(f"Out-of-range SMPL-X vertex IDs: {invalid}")
        metadata = layout.get("layout_metadata", {})
        recorded_hash = (
            metadata.get("model_sha256") if isinstance(metadata, dict) else None
        )
        if recorded_hash and recorded_hash != mesh.sha256:
            raise ValueError(
                "Layout was edited against a different model file: "
                f"recorded SHA-256={recorded_hash}, supplied={mesh.sha256}"
            )
        recorded_count = (
            metadata.get("topology_vertex_count")
            if isinstance(metadata, dict)
            else None
        )
        if recorded_count is not None and int(recorded_count) != vertex_count:
            raise ValueError(
                f"Recorded vertex count {recorded_count} != supplied model {vertex_count}"
            )
    by_vertex: defaultdict[int, list[str]] = defaultdict(list)
    for label, vertex in indices.items():
        if vertex is not None:
            by_vertex[int(vertex)].append(label)
    duplicates = {
        vertex: tuple(marker_labels)
        for vertex, marker_labels in by_vertex.items()
        if len(marker_labels) > 1
    }
    return ValidationResult(labels, assigned, unassigned, duplicates, vertex_count)
