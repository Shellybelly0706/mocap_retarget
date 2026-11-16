#!/usr/bin/env python3
"""Safely initialize, edit, and validate a MoSh++ SMPL-X marker layout.

Only exact labels from MoSh++'s marker database (or an explicit alias in its
``labels_map.py``) are eligible for automatic transfer. All other assignments
must be made by a user with Shift+click on the exact SMPL-X neutral topology
that will be used by MoSh++. Unknown markers remain JSON ``null``.

The editor never infers anatomical locations from marker names or trajectories.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# This editor stores markers strictly on the SMPL-X surface.  A positive
# MoSh++ distance_from_skin would move them outside the mesh along its normals;
# a negative value would move them inside it.
DEFAULT_DISTANCE_FROM_SKIN_M = 0.0
# Matplotlib's 3-D depth sorting can hide a scatter point that is exactly
# coplanar with an opaque triangle.  This offset is display-only; saved marker
# vertex IDs and MoSh++'s distance_from_skin remain exactly on the surface.
DISPLAY_MARKER_OFFSET_M = 0.006


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    path: Path
    sha256: str


@dataclass(frozen=True)
class ValidationResult:
    labels: tuple[str, ...]
    assigned: tuple[str, ...]
    unassigned: tuple[str, ...]
    duplicate_vertices: Mapping[int, tuple[str, ...]]
    vertex_count: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Write valid JSON to a same-directory temporary file, fsync, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def backup_once(path: Path) -> Path | None:
    """Create a byte-for-byte backup once; never replace an earlier backup."""
    if not path.exists():
        return None
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Layout root must be a JSON object: {path}")
    return value


def load_npz_labels(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "labels" not in archive:
            raise ValueError(f"NPZ has no labels field: {path}")
        labels_array = np.asarray(archive["labels"])
    if labels_array.ndim != 1:
        raise ValueError(f"NPZ labels must be one-dimensional, got {labels_array.shape}")
    labels = tuple(str(item) for item in labels_array)
    if len(labels) != len(set(labels)):
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        raise ValueError(f"NPZ labels are not unique: {duplicates}")
    if any(not label.strip() for label in labels):
        raise ValueError("NPZ contains an empty marker label")
    return labels


def load_c3d_labels(
    path: Path, moshpp_root: Path, marker_prefix: str | None = None
) -> tuple[str, ...]:
    """Load C3D point labels with MoSh++'s bundled dependency-free reader."""
    if not path.is_file():
        raise FileNotFoundError(f"C3D does not exist: {path}")
    reader_path = moshpp_root / "src/moshpp/tools/c3d.py"
    if not reader_path.is_file():
        raise FileNotFoundError(f"MoSh++ C3D reader is missing: {reader_path}")
    spec = importlib.util.spec_from_file_location("_moshpp_bundled_c3d", reader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MoSh++ C3D reader: {reader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with path.open("rb") as stream:
        reader = module.Reader(stream)
        raw_labels = reader.point_labels

    labels = []
    wanted_prefix = None
    if marker_prefix:
        wanted_prefix = marker_prefix.rstrip(":") + ":"
    for raw_label in raw_labels:
        if isinstance(raw_label, bytes):
            label = raw_label.decode("utf-8", errors="replace")
        else:
            label = str(raw_label)
        label = label.rstrip("\x00 ").strip()
        if not label or (wanted_prefix and not label.startswith(wanted_prefix)):
            continue
        # MocapSession, which MoSh++ uses for C3D input, removes the subject
        # namespace before matching observations to marker-layout labels.
        if wanted_prefix:
            label = label[len(wanted_prefix) :]
        labels.append(label)
    if not labels:
        detail = f" with prefix {wanted_prefix!r}" if wanted_prefix else ""
        raise ValueError(f"C3D contains no point labels{detail}: {path}")
    if len(labels) != len(set(labels)):
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        raise ValueError(f"C3D point labels are not unique: {duplicates}")
    return tuple(labels)


def literal_assignment(path: Path, variable_name: str) -> Any:
    """Read one literal assignment without importing a dependency-heavy module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in targets):
            value_node = node.value
            if value_node is None:
                break
            return ast.literal_eval(value_node)
    raise ValueError(f"Could not find literal {variable_name!r} in {path}")


def load_official_marker_database(moshpp_root: Path) -> tuple[dict[str, int], Path, Path]:
    """Reproduce MoSh++'s SMPL-X marker database without importing MoSh++."""
    marker_file = moshpp_root / "src/moshpp/marker_layout/marker_vids.py"
    alias_file = moshpp_root / "src/moshpp/marker_layout/labels_map.py"
    mapping_file = moshpp_root / "support_data/smplx_fit2_smplh.npz"
    for path in (marker_file, alias_file, mapping_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required MoSh++ source/support file is missing: {path}")

    base = literal_assignment(marker_file, "all_marker_vids")
    if not isinstance(base, dict) or "smpl" not in base or "smplx" not in base:
        raise ValueError(f"Unexpected all_marker_vids structure in {marker_file}")
    with np.load(mapping_file, allow_pickle=False) as archive:
        if "smh2smhf" not in archive:
            raise ValueError(f"Mapping file has no smh2smhf array: {mapping_file}")
        smplh_to_smplx = np.asarray(archive["smh2smhf"])
    result = {str(label): int(vid) for label, vid in base["smplx"].items()}
    for label, smplh_vid in base["smpl"].items():
        if label in result:
            continue
        if not 0 <= int(smplh_vid) < len(smplh_to_smplx):
            continue
        result[str(label)] = int(smplh_to_smplx[int(smplh_vid)])
    aliases = literal_assignment(alias_file, "general_labels_map")
    if not isinstance(aliases, dict):
        raise ValueError(f"Unexpected general_labels_map structure in {alias_file}")
    for alias, canonical in aliases.items():
        if canonical in result and alias not in result:
            result[str(alias)] = result[str(canonical)]
    return result, marker_file, alias_file


def exact_official_matches(
    labels: Sequence[str], moshpp_root: Path
) -> tuple[dict[str, int], dict[str, dict[str, Any]], Path, Path]:
    """Match exact full labels or exact suffixes after Motive's skeleton namespace."""
    database, marker_file, alias_file = load_official_marker_database(moshpp_root)
    matches: dict[str, int] = {}
    sources: dict[str, dict[str, Any]] = {}
    for label in labels:
        candidates = [label]
        if ":" in label:
            # The prefix is a Motive asset namespace, not part of the marker label.
            candidates.append(label.split(":", 1)[1])
        matched_key = next((candidate for candidate in candidates if candidate in database), None)
        if matched_key is None:
            continue
        matches[label] = int(database[matched_key])
        sources[label] = {
            "method": "official_exact_label",
            "matched_label": matched_key,
            "source": str(marker_file.resolve()),
            "aliases_source": str(alias_file.resolve()),
        }
    return matches, sources, marker_file, alias_file


def make_initial_layout(
    labels: Sequence[str],
    source_path: Path,
    moshpp_root: Path,
    auto_match: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    matches: dict[str, int] = {}
    sources: dict[str, dict[str, Any]] = {}
    marker_file = moshpp_root / "src/moshpp/marker_layout/marker_vids.py"
    alias_file = moshpp_root / "src/moshpp/marker_layout/labels_map.py"
    if auto_match:
        matches, sources, marker_file, alias_file = exact_official_matches(labels, moshpp_root)
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
            "model_path": None,
            "model_sha256": None,
            "topology_vertex_count": None,
            "policy": (
                "Only exact official labels/aliases are auto-filled; all other "
                "vertices require manual surface clicks and may remain null."
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
) -> tuple[dict[str, Any], dict[str, int], Path | None]:
    """Initialize only when absent or non-JSON; preserve any valid JSON layout."""
    if layout_path.exists():
        try:
            existing = read_json(layout_path)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            backup = backup_once(layout_path)
            layout, matches = make_initial_layout(labels, source_path, moshpp_root, auto_match)
            atomic_json_write(layout_path, layout)
            return layout, matches, backup
        return existing, {}, None
    layout, matches = make_initial_layout(labels, source_path, moshpp_root, auto_match)
    atomic_json_write(layout_path, layout)
    return layout, matches, None


def dictionary_value(data: Mapping[Any, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
        encoded = name.encode()
        if encoded in data:
            return data[encoded]
    raise KeyError(f"None of {names} exists in the model")


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "r"):
        value = value.r
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_smplx_mesh(path: Path) -> MeshData:
    """Load neutral template vertices/faces from an NPZ or pickle model."""
    if not path.is_file():
        raise FileNotFoundError(f"SMPL-X neutral model does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            data = {key: archive[key] for key in archive.files}
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            try:
                data = pickle.load(stream, encoding="latin1")
            except TypeError:
                stream.seek(0)
                data = pickle.load(stream)
        if not isinstance(data, Mapping):
            if hasattr(data, "__dict__"):
                data = vars(data)
            else:
                raise ValueError("Pickle model is not a mapping-like SMPL-X model")
    else:
        raise ValueError("SMPL-X model must be .npz, .pkl, or .pickle")

    vertices = as_numpy(dictionary_value(data, "v_template", "vertices", "v")).squeeze()
    faces = as_numpy(dictionary_value(data, "f", "faces")).squeeze()
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected v_template shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected face array shape: {faces.shape}")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not np.all(np.isfinite(vertices)):
        raise ValueError("SMPL-X v_template contains non-finite values")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("SMPL-X faces contain out-of-range vertex IDs")
    return MeshData(vertices, faces, path.resolve(), sha256_file(path))


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
            if vertex is not None and (isinstance(vertex, bool) or not isinstance(vertex, int)):
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
    labels_match = labels == expected if require_label_order else set(labels) == set(expected)
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
        recorded_hash = metadata.get("model_sha256") if isinstance(metadata, dict) else None
        if recorded_hash and recorded_hash != mesh.sha256:
            raise ValueError(
                "Layout was edited against a different model file: "
                f"recorded SHA-256={recorded_hash}, supplied={mesh.sha256}"
            )
        recorded_count = metadata.get("topology_vertex_count") if isinstance(metadata, dict) else None
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


class MarkerLayoutEditor:
    def __init__(self, layout_path: Path, layout: dict[str, Any], mesh: MeshData) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.plt = plt
        self.layout_path = layout_path
        self.layout = layout
        self.mesh = mesh
        self.indices = get_indices(layout)
        self.labels = list(self.indices)
        self.current = self._first_unassigned_index()
        self.undo_stack: list[tuple[str, int | None, Any]] = []
        self.pending_duplicate: tuple[str, int, tuple[str, ...]] | None = None
        self.feedback = ""
        self.overlay_artists: list[Any] = []
        self.visible_marker_count = 0

        # Static triangle data used for exact camera-ray/surface intersection.
        # Picking the old all-vertex scatter could select a vertex on the back
        # of the body through the visible surface.
        triangle_vertices = mesh.vertices[mesh.faces]
        self.triangle_origins = triangle_vertices[:, 0]
        self.triangle_edge1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
        self.triangle_edge2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
        face_normals = np.cross(self.triangle_edge1, self.triangle_edge2)
        self.vertex_normals = np.zeros_like(mesh.vertices)
        for corner in range(3):
            np.add.at(self.vertex_normals, mesh.faces[:, corner], face_normals)
        normal_lengths = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        self.vertex_normals /= np.maximum(normal_lengths, 1e-12)
        self.mesh_center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2

        metadata = self.layout.setdefault("layout_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("layout_metadata must be an object when present")
        existing_hash = metadata.get("model_sha256")
        if existing_hash and existing_hash != mesh.sha256:
            raise ValueError("Refusing to edit a layout created against another model SHA-256")
        metadata.update(
            {
                "model_path": str(mesh.path),
                "model_sha256": mesh.sha256,
                "topology_vertex_count": len(mesh.vertices),
            }
        )
        self.layout.setdefault("marker_sources", {})
        backup_once(self.layout_path)
        self._save()

        self.fig = plt.figure(figsize=(14, 9))
        self.ax = self.fig.add_axes([0.03, 0.06, 0.70, 0.90], projection="3d")
        self.ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.faces,
            mesh.vertices[:, 2],
            color="#b8c5d6",
            alpha=1.0,
            linewidth=0.0,
            edgecolor="none",
            shade=True,
        )
        self.filled_scatter = self.ax.scatter(
            [], [], [], s=70, c="#ff2020", edgecolors="white", linewidths=0.8,
            depthshade=False,
        )
        self.current_scatter = self.ax.scatter(
            [], [], [], s=140, c="#ffe600", edgecolors="black", linewidths=1.0,
            depthshade=False,
        )
        self._equal_axes()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.status_artist = self.fig.text(0.75, 0.50, "", va="top", fontsize=10, wrap=True)
        self.help_artist = self.fig.text(
            0.75,
            0.94,
            "Shift+left-click: assign visible surface\n"
            "Mouse drag/wheel: rotate/zoom\n"
            "N/P: next/previous   S: skip\n"
            "U: undo   Delete/C: clear\n"
            "Y: confirm duplicate vertex\n"
            "Only markers on the visible body surface are shown",
            va="top",
            fontsize=10,
        )
        textbox_ax = self.fig.add_axes([0.75, 0.39, 0.22, 0.045])
        self.textbox = TextBox(textbox_ax, "marker #/name ", initial=str(self.current + 1))
        self.textbox.on_submit(self._select_from_text)
        controls = [
            ("Previous", self.previous),
            ("Next", self.next),
            ("Skip", self.next),
            ("Undo", self.undo),
            ("Clear", self.clear_current),
        ]
        self.buttons = []
        for index, (name, callback) in enumerate(controls):
            button_ax = self.fig.add_axes([0.75 + (index % 2) * 0.115, 0.30 - (index // 2) * 0.055, 0.105, 0.042])
            button = Button(button_ax, name)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("button_press_event", self._on_surface_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_view_changed)
        self.fig.canvas.mpl_connect("scroll_event", self._on_view_changed)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh()

    def _first_unassigned_index(self) -> int:
        for index, label in enumerate(self.labels):
            if self.indices[label] is None:
                return index
        return 0

    def _equal_axes(self) -> None:
        mins = self.mesh.vertices.min(axis=0)
        maxs = self.mesh.vertices.max(axis=0)
        center = (mins + maxs) / 2
        half = max(maxs - mins) / 2 * 1.05
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_box_aspect((1, 1, 1))

    def _save(self) -> None:
        atomic_json_write(self.layout_path, self.layout)

    def _source_for(self, label: str) -> Any:
        return self.layout.get("marker_sources", {}).get(label)

    def _display_points(self, vertex_ids: Sequence[int]) -> np.ndarray:
        """Return visibly raised points without changing saved surface vertices."""
        ids = np.asarray(vertex_ids, dtype=np.int64)
        return (
            self.mesh.vertices[ids]
            + DISPLAY_MARKER_OFFSET_M * self.vertex_normals[ids]
        )

    def _vertex_is_visible(self, vertex_id: int) -> bool:
        """Return whether this surface vertex is the camera ray's first hit."""
        # get_proj() initializes Matplotlib's private camera basis on the first
        # frame; _get_camera_loc() otherwise fails before the initial draw.
        self.ax.get_proj()
        origin = np.asarray(self.ax._get_camera_loc(), dtype=float)
        target = self.mesh.vertices[int(vertex_id)]
        direction = target - origin
        target_distance = float(np.linalg.norm(direction))
        if not np.isfinite(target_distance) or target_distance <= 1e-12:
            return False
        direction /= target_distance

        ray = np.broadcast_to(direction, self.triangle_edge2.shape)
        pvec = np.cross(ray, self.triangle_edge2)
        determinant = np.einsum("ij,ij->i", self.triangle_edge1, pvec)
        candidate = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[candidate] = 1.0 / determinant[candidate]
        tvec = origin - self.triangle_origins
        barycentric_u = inverse * np.einsum("ij,ij->i", tvec, pvec)
        qvec = np.cross(tvec, self.triangle_edge1)
        barycentric_v = inverse * np.einsum("ij,ij->i", ray, qvec)
        distance = inverse * np.einsum("ij,ij->i", self.triangle_edge2, qvec)
        epsilon = 1e-8
        candidate &= barycentric_u >= -epsilon
        candidate &= barycentric_v >= -epsilon
        candidate &= barycentric_u + barycentric_v <= 1.0 + epsilon
        candidate &= distance > epsilon
        hits = distance[candidate]
        if not len(hits):
            return False

        # A small tolerance handles a marker vertex shared by several triangles
        # and floating-point error. Rear-side markers are separated from the
        # first hit by the body's thickness and therefore remain hidden.
        first_hit_distance = float(np.min(hits))
        return target_distance - first_hit_distance <= 0.008

    def _update_marker_overlay(self) -> None:
        """Overlay only markers lying on the camera-visible body surface."""
        from matplotlib import patheffects
        from mpl_toolkits.mplot3d import proj3d

        for artist in self.overlay_artists:
            artist.remove()
        self.overlay_artists.clear()

        assigned = [label for label in self.labels if self.indices[label] is not None]
        visible = [
            label
            for label in assigned
            if self._vertex_is_visible(int(self.indices[label]))
        ]
        self.visible_marker_count = len(visible)
        if not visible:
            return
        points = self._display_points([int(self.indices[label]) for label in visible])
        projection = self.ax.get_proj()
        for label, point in zip(visible, points):
            screen_x, screen_y, _ = proj3d.proj_transform(*point, projection)
            current = label == self.labels[self.current]
            dot_color = "#ffe600" if current else "#ff2020"
            label_color = "#111111" if current else "#c00000"
            dot = self.ax.text2D(
                screen_x,
                screen_y,
                "●",
                transform=self.ax.transData,
                color=dot_color,
                fontsize=13 if current else 10,
                fontweight="bold",
                zorder=10_000,
                clip_on=False,
            )
            dot.set_path_effects(
                [patheffects.withStroke(linewidth=2.0, foreground="black" if current else "white")]
            )
            annotation = self.ax.annotate(
                label,
                xy=(screen_x, screen_y),
                xycoords=self.ax.transData,
                xytext=(5, 4),
                textcoords="offset points",
                color=label_color,
                fontsize=8,
                fontweight="bold",
                zorder=10_001,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "#fff176" if current else "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
            self.overlay_artists.extend((dot, annotation))

    def _on_view_changed(self, event: Any) -> None:
        if event.inaxes is self.ax:
            self._update_marker_overlay()
            self.fig.canvas.draw_idle()

    def _refresh(self) -> None:
        # Marker dots and labels are rendered by the visibility-tested 2-D
        # overlay below. Keeping the 3-D collections empty avoids Matplotlib's
        # collection depth sorting from revealing markers through the body.
        empty = np.array([])
        self.filled_scatter._offsets3d = (empty, empty, empty)
        label = self.labels[self.current]
        vertex = self.indices[label]
        self.current_scatter._offsets3d = (empty, empty, empty)
        assigned_count = sum(value is not None for value in self.indices.values())
        pending = ""
        if self.pending_duplicate:
            _, candidate, conflicts = self.pending_duplicate
            pending = (
                f"\n\nDUPLICATE PENDING: vertex {candidate} is already used by "
                f"{', '.join(conflicts)}. Press Y to confirm, or pick another vertex."
            )
        self.status_artist.set_text(
            f"Current {self.current + 1}/{len(self.labels)}\n{label}\n"
            f"vertex: {vertex}\nsource: {self._source_for(label)}\n\n"
            f"progress: {assigned_count}/{len(self.labels)} assigned{pending}"
            f"{self.feedback}"
        )
        self.ax.set_title(f"SMPL-X marker editor — {label}")
        self._update_marker_overlay()
        self.fig.canvas.draw_idle()

    def _select_from_text(self, text: str) -> None:
        value = text.strip()
        if not value:
            return
        try:
            index = int(value) - 1
        except ValueError:
            matches = [index for index, label in enumerate(self.labels) if label == value]
            if not matches:
                matches = [index for index, label in enumerate(self.labels) if label.endswith(value)]
            if len(matches) != 1:
                self.status_artist.set_text(f"Could not select unique marker: {value!r}")
                self.fig.canvas.draw_idle()
                return
            index = matches[0]
        if 0 <= index < len(self.labels):
            self.current = index
            self.pending_duplicate = None
            self._refresh()

    def _visible_surface_vertex(self, event: Any) -> int | None:
        """Ray-cast through the click and return a vertex on the first hit face."""
        from mpl_toolkits.mplot3d import proj3d

        projected_xy = self.ax.transData.inverted().transform([event.x, event.y])
        projection = self.ax.get_proj()
        _, _, center_z = proj3d.proj_transform(*self.mesh_center, projection)
        target = np.asarray(
            proj3d.inv_transform(
                projected_xy[0],
                projected_xy[1],
                center_z,
                np.linalg.inv(projection),
            ),
            dtype=float,
        ).reshape(3)
        origin = np.asarray(self.ax._get_camera_loc(), dtype=float)
        direction = target - origin
        direction_norm = np.linalg.norm(direction)
        if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
            return None
        direction /= direction_norm

        # Vectorized Moller-Trumbore ray/triangle intersection. The minimum
        # positive ray distance is the visible surface, so rear faces can never
        # be selected through the body.
        ray = np.broadcast_to(direction, self.triangle_edge2.shape)
        pvec = np.cross(ray, self.triangle_edge2)
        determinant = np.einsum("ij,ij->i", self.triangle_edge1, pvec)
        candidate = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[candidate] = 1.0 / determinant[candidate]
        tvec = origin - self.triangle_origins
        barycentric_u = inverse * np.einsum("ij,ij->i", tvec, pvec)
        qvec = np.cross(tvec, self.triangle_edge1)
        barycentric_v = inverse * np.einsum("ij,ij->i", ray, qvec)
        distance = inverse * np.einsum("ij,ij->i", self.triangle_edge2, qvec)
        epsilon = 1e-9
        candidate &= barycentric_u >= -epsilon
        candidate &= barycentric_v >= -epsilon
        candidate &= barycentric_u + barycentric_v <= 1.0 + epsilon
        candidate &= distance > epsilon
        hits = np.flatnonzero(candidate)
        if not len(hits):
            return None
        triangle_index = int(hits[np.argmin(distance[hits])])
        intersection = origin + distance[triangle_index] * direction
        face_vertices = self.mesh.faces[triangle_index]
        nearest = np.argmin(
            np.linalg.norm(self.mesh.vertices[face_vertices] - intersection, axis=1)
        )
        return int(face_vertices[nearest])

    def _on_surface_click(self, event: Any) -> None:
        key = (event.key or "").lower()
        if event.inaxes is not self.ax or event.button != 1 or "shift" not in key:
            return
        vertex = self._visible_surface_vertex(event)
        if vertex is None:
            self.feedback = "\n\nNo visible body surface under the click."
            self._refresh()
            return
        self.feedback = ""
        label = self.labels[self.current]
        conflicts = tuple(
            other
            for other, assigned_vertex in self.indices.items()
            if other != label and assigned_vertex == vertex
        )
        if conflicts:
            self.pending_duplicate = (label, vertex, conflicts)
            self._refresh()
            return
        self._assign(label, vertex, duplicate_confirmed=False)

    def _assign(self, label: str, vertex: int, duplicate_confirmed: bool) -> None:
        old_vertex = self.indices[label]
        old_source = self._source_for(label)
        self.undo_stack.append((label, old_vertex, old_source))
        self.indices[label] = int(vertex)
        set_index(self.layout, label, int(vertex))
        source = {
            "method": "manual_surface_click",
            "model_path": str(self.mesh.path),
            "model_sha256": self.mesh.sha256,
            "duplicate_vertex_confirmed": bool(duplicate_confirmed),
        }
        self.layout["marker_sources"][label] = source
        self.pending_duplicate = None
        self._save()
        self.next()

    def next(self, _event: Any = None) -> None:
        self.current = (self.current + 1) % len(self.labels)
        self.pending_duplicate = None
        self._refresh()

    def previous(self, _event: Any = None) -> None:
        self.current = (self.current - 1) % len(self.labels)
        self.pending_duplicate = None
        self._refresh()

    def clear_current(self, _event: Any = None) -> None:
        label = self.labels[self.current]
        old_vertex = self.indices[label]
        old_source = self._source_for(label)
        if old_vertex is None:
            return
        self.undo_stack.append((label, old_vertex, old_source))
        self.indices[label] = None
        set_index(self.layout, label, None)
        self.layout["marker_sources"].pop(label, None)
        self.pending_duplicate = None
        self._save()
        self._refresh()

    def undo(self, _event: Any = None) -> None:
        if not self.undo_stack:
            return
        label, old_vertex, old_source = self.undo_stack.pop()
        self.current = self.labels.index(label)
        self.indices[label] = old_vertex
        set_index(self.layout, label, old_vertex)
        if old_source is None:
            self.layout["marker_sources"].pop(label, None)
        else:
            self.layout["marker_sources"][label] = old_source
        self.pending_duplicate = None
        self._save()
        self._refresh()

    def _on_key(self, event: Any) -> None:
        key = event.key or ""
        if key.lower() == "n" or key == "right":
            self.next()
        elif key.lower() == "p" or key == "left":
            self.previous()
        elif key.lower() == "s":
            self.next()
        elif key.lower() == "u":
            self.undo()
        elif key.lower() == "c" or key == "delete":
            self.clear_current()
        elif key.lower() == "y" and self.pending_duplicate:
            label, vertex, _ = self.pending_duplicate
            self._assign(label, vertex, duplicate_confirmed=True)

    def show(self) -> None:
        self.plt.show()


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
    manual = [
        label
        for label in result.assigned
        if isinstance(sources.get(label), dict)
        and sources[label].get("method") == "manual_surface_click"
    ]
    other = [label for label in result.assigned if label not in automatic and label not in manual]
    print("\n=== SMPL-X marker layout validation ===")
    order_note = "exact order" if source_kind == "NPZ" else "set match; C3D order is ignored"
    print(f"labels: {len(result.labels)} ({source_kind} label {order_note})")
    print(f"vertex topology size: {result.vertex_count if result.vertex_count is not None else 'not available'}")
    print(f"automatic exact matches: {len(automatic)}")
    print(f"manual assignments: {len(manual)}")
    print(f"assigned without provenance: {len(other)}")
    print(f"unassigned/null: {len(result.unassigned)}")
    if automatic:
        print("Automatic: " + ", ".join(automatic))
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True, help="Target MoSh++ marker-layout JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path, help="Marker NPZ supplying the exact labels")
    source.add_argument("--c3d", type=Path, help="Motive C3D supplying point labels")
    parser.add_argument(
        "--marker-prefix",
        help=(
            "For C3D input, keep only labels under this namespace "
            "and strip it as MoSh++ does (for example Skeleton_Zixuan); "
            "ignored for NPZ"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Exact neutral SMPL-X .pkl/.npz that MoSh++ will use",
    )
    parser.add_argument(
        "--moshpp-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "moshpp",
        help="MoSh++ repository root (default: sibling ../moshpp)",
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Create/repair the JSON and validate it without opening the editor",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing JSON without opening the editor",
    )
    parser.add_argument(
        "--no-auto-match",
        action="store_true",
        help="Disable even exact official label/alias transfer during initialization",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout_path = args.layout.expanduser().resolve()
    moshpp_root = args.moshpp_root.expanduser().resolve()
    if args.npz is not None:
        source_path = args.npz.expanduser().resolve()
        source_kind = "NPZ"
        labels = load_npz_labels(source_path)
    else:
        source_path = args.c3d.expanduser().resolve()
        source_kind = "C3D"
        labels = load_c3d_labels(source_path, moshpp_root, args.marker_prefix)

    if args.validate_only:
        layout = read_json(layout_path)
        matches: dict[str, int] = {}
        backup = None
    else:
        layout, matches, backup = initialize_layout(
            layout_path,
            labels,
            source_path,
            moshpp_root,
            auto_match=not args.no_auto_match,
        )
    if backup is not None:
        print(f"Preserved original bytes at: {backup}")
    if matches:
        print(f"Applied {len(matches)} exact official label/alias match(es).")
    else:
        print("Applied 0 exact official label/alias matches.")

    mesh = load_smplx_mesh(args.model.expanduser().resolve()) if args.model else None
    result = validate_layout(
        layout,
        labels,
        mesh,
        require_label_order=source_kind == "NPZ",
    )
    report(result, layout, source_kind)

    if args.initialize_only or args.validate_only:
        return 0
    if mesh is None:
        raise ValueError(
            "Interactive editing requires --model pointing to the exact neutral SMPL-X "
            "model that MoSh++ will use. Uncertain markers remain null."
        )
    editor = MarkerLayoutEditor(layout_path, layout, mesh)
    editor.show()
    # Reload after the GUI closes so the final report reflects saved progress.
    final_layout = read_json(layout_path)
    report(
        validate_layout(
            final_layout,
            labels,
            mesh,
            require_label_order=source_kind == "NPZ",
        ),
        final_layout,
        source_kind,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
