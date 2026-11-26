"""Reusable avatars gltf helpers."""

from __future__ import annotations

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
