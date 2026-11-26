#!/usr/bin/env python3
"""Convert the CesiumMan glTF skin to a compact upper-body NumPy asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPONENT_DTYPES = {
    5121: np.dtype("u1"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
UPPER_BODY_JOINT_COUNT = 11


def read_accessor(document: dict[str, object], buffer: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    components = TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * components)
    values = np.ndarray(
        shape=(count, components),
        dtype=dtype,
        buffer=buffer,
        offset=offset,
        strides=(stride, dtype.itemsize),
    )
    return values.copy()


def convert(source: Path, output: Path) -> None:
    document = json.loads(source.read_text())
    binary_path = source.parent / document["buffers"][0]["uri"]
    binary = binary_path.read_bytes()
    primitive = document["meshes"][0]["primitives"][0]

    vertices = read_accessor(document, binary, primitive["attributes"]["POSITION"])
    texcoords = read_accessor(document, binary, primitive["attributes"]["TEXCOORD_0"])
    joint_ids = read_accessor(document, binary, primitive["attributes"]["JOINTS_0"])
    joint_weights = read_accessor(
        document, binary, primitive["attributes"]["WEIGHTS_0"]
    )
    faces = read_accessor(document, binary, primitive["indices"]).reshape(-1, 3)

    skin = document["skins"][0]
    joint_nodes = skin["joints"][:UPPER_BODY_JOINT_COUNT]
    joint_names = np.asarray([document["nodes"][node]["name"] for node in joint_nodes])
    inverse_bind = read_accessor(document, binary, skin["inverseBindMatrices"])
    inverse_bind = inverse_bind.reshape(-1, 4, 4).transpose(0, 2, 1)
    bind_matrices = np.linalg.inv(inverse_bind[:UPPER_BODY_JOINT_COUNT])
    bind_positions = bind_matrices[:, :3, 3].astype(np.float32)
    bind_rotations = bind_matrices[:, :3, :3].astype(np.float32)

    weights = np.zeros((len(vertices), UPPER_BODY_JOINT_COUNT), dtype=np.float32)
    for influence in range(4):
        ids = joint_ids[:, influence]
        values = joint_weights[:, influence]
        for joint in range(UPPER_BODY_JOINT_COUNT):
            selected = ids == joint
            weights[selected, joint] += values[selected]

    upper_weight = weights.sum(axis=1)
    # Remove lower-body triangles instead of amplifying tiny torso weights at
    # the hip seam when the retained weights are normalized.
    keep_faces = np.all(upper_weight[faces] > 0.50, axis=1)
    faces = faces[keep_faces]
    used_vertices = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)

    vertices = vertices[used_vertices].astype(np.float32)
    texcoords = texcoords[used_vertices].astype(np.float32)
    weights = weights[used_vertices]
    weights /= weights.sum(axis=1, keepdims=True)
    faces = remap[faces].astype(np.int32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=vertices,
        texcoords=texcoords,
        faces=faces,
        weights=weights,
        joint_names=joint_names,
        bind_positions=bind_positions,
        bind_rotations=bind_rotations,
    )
    print(
        f"Wrote {output}: {len(vertices)} vertices, {len(faces)} faces, "
        f"{len(joint_names)} upper-body joints"
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="CesiumMan.gltf input")
    parser.add_argument("output", type=Path, help="upper_body_skin.npz output")
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    convert(arguments.source, arguments.output)
