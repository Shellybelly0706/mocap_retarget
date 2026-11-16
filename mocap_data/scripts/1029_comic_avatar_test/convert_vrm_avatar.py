#!/usr/bin/env python3
"""Convert a humanoid VRM file into a compact MuJoCo skin."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


COMPONENT_DTYPES = {
    5121: np.dtype("u1"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
HUMANOID_BONES = (
    "hips",
    "spine",
    "chest",
    "upperChest",
    "neck",
    "head",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "leftToes",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "rightToes",
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    "leftThumbProximal",
    "leftThumbIntermediate",
    "leftThumbDistal",
    "leftIndexProximal",
    "leftIndexIntermediate",
    "leftIndexDistal",
    "leftMiddleProximal",
    "leftMiddleIntermediate",
    "leftMiddleDistal",
    "leftRingProximal",
    "leftRingIntermediate",
    "leftRingDistal",
    "leftLittleProximal",
    "leftLittleIntermediate",
    "leftLittleDistal",
    "rightThumbProximal",
    "rightThumbIntermediate",
    "rightThumbDistal",
    "rightIndexProximal",
    "rightIndexIntermediate",
    "rightIndexDistal",
    "rightMiddleProximal",
    "rightMiddleIntermediate",
    "rightMiddleDistal",
    "rightRingProximal",
    "rightRingIntermediate",
    "rightRingDistal",
    "rightLittleProximal",
    "rightLittleIntermediate",
    "rightLittleDistal",
)
LEG_BONES = {
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "leftToes",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
    "rightToes",
}
UPPER_BODY_BONES = tuple(name for name in HUMANOID_BONES if name not in LEG_BONES)
CORE_UPPER_BODY_BONES = (
    "hips",
    "spine",
    "chest",
    "upperChest",
    "neck",
    "head",
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
)

# VRM/glTF is Y-up.  The visualizer and MuJoCo scene use Z-up.
Y_UP_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def read_glb(path: Path) -> tuple[dict[str, object], bytes]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path} is too short to be a GLB/VRM file")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError(f"{path} is not a valid glTF 2.0 binary file")

    offset = 12
    json_document: dict[str, object] | None = None
    binary = b""
    while offset < len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if chunk_type == 0x4E4F534A:  # JSON
            json_document = json.loads(chunk.rstrip(b" \t\r\n\0"))
        elif chunk_type == 0x004E4942:  # BIN
            binary = chunk
        offset += 8 + length
    if json_document is None or not binary:
        raise ValueError(f"{path} must contain JSON and BIN chunks")
    return json_document, binary


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
    ).copy()
    if accessor.get("normalized"):
        if np.issubdtype(dtype, np.unsignedinteger):
            values = values.astype(np.float32) / np.iinfo(dtype).max
        else:
            limit = np.iinfo(dtype).max
            values = np.maximum(values.astype(np.float32) / limit, -1.0)
    return values


def embedded_image(
    document: dict[str, object], buffer: bytes, material_index: int
) -> tuple[bytes, str]:
    material = document["materials"][material_index]
    texture_info = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if texture_info is None:
        raise ValueError("The avatar material does not define a base-color texture")
    texture = document["textures"][texture_info["index"]]
    image = document["images"][texture["source"]]
    if "bufferView" not in image:
        raise ValueError("Only VRMs with an embedded texture are supported")
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    payload = buffer[start : start + view["byteLength"]]
    mime_type = image.get("mimeType", "image/png")
    extension = {"image/png": ".png", "image/jpeg": ".jpg"}.get(mime_type)
    if extension is None:
        raise ValueError(f"Unsupported embedded texture type: {mime_type}")
    return payload, extension


def convert(
    source: Path,
    output: Path,
    texture_output: Path | None = None,
    upper_body: bool = False,
) -> None:
    document, binary = read_glb(source)
    vrm = document.get("extensions", {}).get("VRM")
    if vrm is None:
        raise ValueError(f"{source} does not contain the VRM 0.x humanoid extension")

    humanoid = {
        entry["bone"]: entry["node"]
        for entry in vrm.get("humanoid", {}).get("humanBones", [])
    }
    requested_bones = UPPER_BODY_BONES if upper_body else HUMANOID_BONES
    selected_bones = tuple(name for name in requested_bones if name in humanoid)
    required_bones = CORE_UPPER_BODY_BONES if upper_body else HUMANOID_BONES
    missing = [name for name in required_bones if name not in humanoid]
    if missing:
        raise ValueError(f"VRM is missing required humanoid bones: {missing}")

    mesh_index = next(
        (node["mesh"] for node in document["nodes"] if "mesh" in node), None
    )
    if mesh_index is None:
        raise ValueError("VRM does not contain a skinned mesh")
    primitives = document["meshes"][mesh_index]["primitives"]
    if len(primitives) != 1:
        raise ValueError("Only single-primitive VRM meshes are currently supported")
    primitive = primitives[0]
    required_attributes = {"POSITION", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    missing_attributes = required_attributes.difference(primitive["attributes"])
    if missing_attributes:
        raise ValueError(f"VRM mesh is missing attributes: {sorted(missing_attributes)}")

    vertices = read_accessor(
        document, binary, primitive["attributes"]["POSITION"]
    ).astype(np.float64)
    texcoords = read_accessor(
        document, binary, primitive["attributes"]["TEXCOORD_0"]
    )
    joint_ids = read_accessor(document, binary, primitive["attributes"]["JOINTS_0"])
    joint_weights = read_accessor(
        document, binary, primitive["attributes"]["WEIGHTS_0"]
    ).astype(np.float32)
    faces = read_accessor(document, binary, primitive["indices"]).reshape(-1, 3)

    skin = document["skins"][0]
    skin_joint_nodes = skin["joints"]
    node_to_skin_joint = {node: index for index, node in enumerate(skin_joint_nodes)}
    selected_skin_joints = [
        node_to_skin_joint[humanoid[name]] for name in selected_bones
    ]

    inverse_bind = read_accessor(
        document, binary, skin["inverseBindMatrices"]
    ).reshape(-1, 4, 4).transpose(0, 2, 1)
    bind_matrices = np.linalg.inv(inverse_bind[selected_skin_joints])

    transform = np.eye(4)
    transform[:3, :3] = Y_UP_TO_Z_UP
    inverse_transform = np.linalg.inv(transform)
    bind_matrices = transform @ bind_matrices @ inverse_transform
    vertices = vertices @ Y_UP_TO_Z_UP.T

    node_parents: dict[int, int] = {}
    for parent, node in enumerate(document["nodes"]):
        for child in node.get("children", []):
            node_parents[child] = parent
    selected_node_to_output = {
        humanoid[name]: index for index, name in enumerate(selected_bones)
    }
    skin_joint_to_output: dict[int, int] = {}
    for skin_joint, node in enumerate(skin_joint_nodes):
        ancestor = node
        while ancestor not in selected_node_to_output:
            if ancestor not in node_parents:
                break
            ancestor = node_parents[ancestor]
        if ancestor in selected_node_to_output:
            skin_joint_to_output[skin_joint] = selected_node_to_output[ancestor]

    weights = np.zeros((len(vertices), len(selected_bones)), dtype=np.float32)
    upper_weight = np.zeros(len(vertices), dtype=np.float32)
    humanoid_node_to_name = {node: name for name, node in humanoid.items()}
    for influence in range(joint_ids.shape[1]):
        ids = joint_ids[:, influence]
        values = joint_weights[:, influence]
        for skin_joint, output_joint in skin_joint_to_output.items():
            selected = ids == skin_joint
            weights[selected, output_joint] += values[selected]
        if upper_body:
            for skin_joint, node in enumerate(skin_joint_nodes):
                ancestor = node
                while ancestor not in humanoid_node_to_name:
                    if ancestor not in node_parents:
                        break
                    ancestor = node_parents[ancestor]
                bone_name = humanoid_node_to_name.get(ancestor)
                if bone_name is not None and bone_name not in LEG_BONES:
                    upper_weight[ids == skin_joint] += values[ids == skin_joint]

    if upper_body:
        keep_faces = np.all(upper_weight[faces] > 0.50, axis=1)
        faces = faces[keep_faces]
        used_vertices = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)
        vertices = vertices[used_vertices]
        texcoords = texcoords[used_vertices]
        weights = weights[used_vertices]
        faces = remap[faces]

    unweighted = weights.sum(axis=1) < 1e-6
    if np.any(unweighted):
        raise ValueError(f"{int(unweighted.sum())} mesh vertices have no humanoid weight")
    weights /= weights.sum(axis=1, keepdims=True)
    vertices = vertices.astype(np.float32)
    texcoords = texcoords.astype(np.float32)
    faces = faces.astype(np.int32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices=vertices,
        texcoords=texcoords,
        faces=faces,
        weights=weights,
        joint_names=np.asarray(selected_bones),
        bind_positions=bind_matrices[:, :3, 3].astype(np.float32),
        bind_rotations=bind_matrices[:, :3, :3].astype(np.float32),
    )

    texture, texture_extension = embedded_image(
        document, binary, primitive.get("material", 0)
    )
    if texture_output is None:
        texture_output = output.with_name(f"avatar_texture{texture_extension}")
    if texture_output.suffix.lower() != texture_extension:
        raise ValueError(
            f"Texture output must use the embedded image extension {texture_extension}"
        )
    texture_output.write_bytes(texture)
    print(
        f"Wrote {output}: {len(vertices)} vertices, {len(faces)} faces, "
        f"{len(selected_bones)} humanoid joints"
    )
    print(f"Wrote {texture_output}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="VRM 0.x input")
    parser.add_argument("output", type=Path, help="converted avatar .npz output")
    parser.add_argument(
        "--texture-output",
        type=Path,
        help="embedded texture output (default: avatar_texture next to output)",
    )
    parser.add_argument(
        "--upper-body",
        action="store_true",
        help="remove leg-weighted triangles and export upper-body joints only",
    )
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    convert(
        arguments.source,
        arguments.output,
        arguments.texture_output,
        arguments.upper_body,
    )
