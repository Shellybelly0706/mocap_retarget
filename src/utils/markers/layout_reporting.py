"""Reusable markers layout reporting helpers."""

from __future__ import annotations

from typing import Any, Mapping

from utils.markers.layout_models import (
    ValidationResult,
)


def report(
    result: ValidationResult,
    layout: Mapping[str, Any],
    source_kind: str,
) -> None:
    sources = layout.get("marker_sources", {})
    automatic = [
        label
        for label in result.assigned
        if isinstance(sources.get(label), dict)
        and sources[label].get("method") == "official_exact_label"
    ]
    referenced = [
        label
        for label in result.assigned
        if isinstance(sources.get(label), dict)
        and sources[label].get("method") == "reference_layout_exact_label"
    ]
    equivalent = [
        label
        for label in result.assigned
        if isinstance(sources.get(label), dict)
        and sources[label].get("method") == "reference_layout_equivalent_label"
    ]
    manual = [
        label
        for label in result.assigned
        if isinstance(sources.get(label), dict)
        and sources[label].get("method") == "manual_surface_click"
    ]
    other = [
        label
        for label in result.assigned
        if label not in automatic
        and label not in referenced
        and label not in equivalent
        and label not in manual
    ]
    print("\n=== SMPL-X marker layout validation ===")
    order_note = (
        "exact order" if source_kind == "NPZ" else "set match; C3D order is ignored"
    )
    print(f"labels: {len(result.labels)} ({source_kind} label {order_note})")
    print(
        f"vertex topology size: {result.vertex_count if result.vertex_count is not None else 'not available'}"
    )
    print(f"automatic exact matches: {len(automatic)}")
    print(f"reference-layout exact matches: {len(referenced)}")
    print(f"reference-layout equivalent-label matches: {len(equivalent)}")
    print(f"manual assignments: {len(manual)}")
    print(f"assigned without provenance: {len(other)}")
    print(f"unassigned/null: {len(result.unassigned)}")
    if automatic:
        print("Automatic: " + ", ".join(automatic))
    if referenced:
        print("Reference layout: " + ", ".join(referenced))
    if equivalent:
        print("Equivalent labels: " + ", ".join(equivalent))
    if manual:
        print("Manual: " + ", ".join(manual))
    if other:
        print("WARNING — assigned without provenance: " + ", ".join(other))
    if result.unassigned:
        print("Unassigned: " + ", ".join(result.unassigned))
    if result.duplicate_vertices:
        print("Duplicate vertex IDs requiring manual confirmation:")
        for vertex, labels in sorted(result.duplicate_vertices.items()):
            print(f"  {vertex}: {', '.join(labels)}")
    else:
        print("Duplicate vertex IDs: none")
