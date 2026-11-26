"""Reusable markers layout io helpers."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.markers.layout_models import (
    MeshData,
)


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
        raise ValueError(
            f"NPZ labels must be one-dimensional, got {labels_array.shape}"
        )
    labels = tuple(str(item) for item in labels_array)
    if len(labels) != len(set(labels)):
        duplicates = sorted(
            label for label, count in Counter(labels).items() if count > 1
        )
        raise ValueError(f"NPZ labels are not unique: {duplicates}")
    if any(not label.strip() for label in labels):
        raise ValueError("NPZ contains an empty marker label")
    return labels


def load_c3d_labels(
    path: Path,
    moshpp_root: Path,
    marker_prefix: str | Sequence[str] | None = None,
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
    # The bundled reader still uses the removed ``np.bool`` alias. Keep the
    # compatibility shim scoped to reader construction so this editor also
    # works with current NumPy releases without modifying MoSh++ itself.
    had_numpy_bool = "bool" in np.__dict__
    if not had_numpy_bool:
        setattr(np, "bool", np.bool_)
    try:
        with path.open("rb") as stream:
            reader = module.Reader(stream)
            raw_labels = reader.point_labels
    finally:
        if not had_numpy_bool:
            delattr(np, "bool")

    marker_prefixes = marker_prefix
    if isinstance(marker_prefixes, str):
        marker_prefixes = (marker_prefixes,)
    wanted_prefixes = tuple(
        prefix.rstrip(":") + ":" for prefix in (marker_prefixes or ())
    )
    if len(wanted_prefixes) != len(set(wanted_prefixes)):
        raise ValueError(f"C3D marker prefixes must be unique: {wanted_prefixes}")

    labels = []
    source_labels: dict[str, str] = {}
    for raw_label in raw_labels:
        if isinstance(raw_label, bytes):
            label = raw_label.decode("utf-8", errors="replace")
        else:
            label = str(raw_label)
        label = label.rstrip("\x00 ").strip()
        if not label:
            continue
        matched_prefix = next(
            (prefix for prefix in wanted_prefixes if label.startswith(prefix)), None
        )
        if wanted_prefixes and matched_prefix is None:
            continue
        # MocapSession, which MoSh++ uses for C3D input, removes the subject
        # namespace before matching observations to marker-layout labels.
        original_label = label
        if matched_prefix is not None:
            label = label[len(matched_prefix) :]
        if label in source_labels:
            if source_labels[label] == original_label:
                raise ValueError(f"C3D point labels are not unique: {[original_label]}")
            # Multiple selected Motive assets may contain the same suffix
            # (for example body and hand assets both containing RWristIn).
            # MoSh++ strips both namespaces, so the layout must contain this
            # normalized marker name exactly once.
            continue
        source_labels[label] = original_label
        labels.append(label)
    if not labels:
        detail = f" with prefixes {wanted_prefixes!r}" if wanted_prefixes else ""
        raise ValueError(f"C3D contains no point labels{detail}: {path}")
    return tuple(labels)


def literal_assignment(path: Path, variable_name: str) -> Any:
    """Read one literal assignment without importing a dependency-heavy module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in targets
        ):
            value_node = node.value
            if value_node is None:
                break
            return ast.literal_eval(value_node)
    raise ValueError(f"Could not find literal {variable_name!r} in {path}")


def load_official_marker_database(
    moshpp_root: Path,
) -> tuple[dict[str, int], Path, Path]:
    """Reproduce MoSh++'s SMPL-X marker database without importing MoSh++."""
    marker_file = moshpp_root / "src/moshpp/marker_layout/marker_vids.py"
    alias_file = moshpp_root / "src/moshpp/marker_layout/labels_map.py"
    mapping_file = moshpp_root / "support_data/smplx_fit2_smplh.npz"
    for path in (marker_file, alias_file, mapping_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required MoSh++ source/support file is missing: {path}"
            )

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
        matched_key = next(
            (candidate for candidate in candidates if candidate in database), None
        )
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
