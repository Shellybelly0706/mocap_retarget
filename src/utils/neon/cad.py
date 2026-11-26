"""Convert the official Neon reference assembly to binary STL meshes.

OpenCascade (cadquery-ocp) is needed only to regenerate the meshes. Loading
the shipped STL files in MuJoCo does not need a CAD library.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def write_binary_stl(path: Path, triangles: np.ndarray) -> int:
    """Write metre-valued triangle vertices, excluding degenerate faces."""
    triangles = np.asarray(triangles, dtype=np.float32)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 2e-14
    triangles, normals, lengths = triangles[keep], normals[keep], lengths[keep]
    normals /= lengths[:, None]
    records = np.zeros(
        len(triangles),
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    records["normal"], records["vertices"] = normals, triangles
    with path.open("wb") as stream:
        stream.write(
            b"Pupil Labs Neon reference; vertex units: metres".ljust(80, b"\0")
        )
        stream.write(struct.pack("<I", len(records)))
        stream.write(records.tobytes())
    return len(records)


def convert_reference_cad(
    source: Path,
    output_dir: Path,
    *,
    linear_deflection_mm: float = 0.08,
    angular_deflection_rad: float = 0.15,
) -> dict:
    """Tessellate the official assembly without changing component placement.

    Input STEP lengths are millimetres; exported STL vertices are metres.
    The camera origin is the CAD lens-opening centre, a reference pose rather
    than a calibrated optical-centre-to-frame transform.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.GeomAbs import GeomAbs_Circle
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    source, output_dir = Path(source), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError(f"Cannot read STEP file: {source}")
    reader.TransferRoots()
    shape = reader.OneShape()
    mesher = BRepMesh_IncrementalMesh(
        shape, linear_deflection_mm, False, angular_deflection_rad, False
    )
    if not mesher.IsDone():
        raise RuntimeError("OpenCascade tessellation failed")

    groups = {"frame": [], "module": [], "hardware": []}
    bounds = []
    camera_candidates = []
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while solids.More():
        solid = solids.Current()
        box = Bnd_Box()
        BRepBndLib.Add_s(solid, box)
        limits = np.array(box.Get()).reshape(2, 3)
        bounds.append(limits)
        dimensions = limits[1] - limits[0]
        if dimensions.max() > 100:
            group = "frame"
        elif dimensions[0] > 35 and dimensions[1] > 25 and dimensions[2] > 15:
            group = "module"
        else:
            group = "hardware"

        faces = TopExp_Explorer(solid, TopAbs_FACE)
        while faces.More():
            face = TopoDS.Face_s(faces.Current())
            location = TopLoc_Location()
            mesh = BRep_Tool.Triangulation_s(face, location)
            if mesh is not None:
                transform = location.Transformation()
                vertices = np.array(
                    [
                        mesh.Node(i).Transformed(transform).Coord()
                        for i in range(1, mesh.NbNodes() + 1)
                    ]
                )
                for i in range(1, mesh.NbTriangles() + 1):
                    indices = np.array(mesh.Triangle(i).Get()) - 1
                    if face.Orientation() == TopAbs_REVERSED:
                        indices = indices[[0, 2, 1]]
                    groups[group].append(vertices[indices] * 0.001)
            faces.Next()

        if group == "module":
            edges = TopExp_Explorer(solid, TopAbs_EDGE)
            while edges.More():
                curve = BRepAdaptor_Curve(TopoDS.Edge_s(edges.Current()))
                if curve.GetType() == GeomAbs_Circle:
                    circle = curve.Circle()
                    centre, axis = circle.Location(), circle.Axis().Direction()
                    if (
                        abs(centre.X()) < 0.5
                        and abs(circle.Radius() - 3.45) < 0.05
                        and axis.Z() > 0.9
                    ):
                        camera_candidates.append((centre.Coord(), axis.Coord()))
                edges.Next()
        solid_count += 1
        solids.Next()

    if not camera_candidates or any(not triangles for triangles in groups.values()):
        raise ValueError(
            "Unexpected Neon CAD structure: missing camera aperture or mesh component"
        )
    centre, forward = max(camera_candidates, key=lambda candidate: candidate[0][2])
    forward = np.asarray(forward)
    up = np.array([0.0, 1.0, 0.0])
    up -= np.dot(up, forward) * forward
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    camera_rotation = np.column_stack((right, -up, forward))
    components = {}
    for group, triangles in groups.items():
        # MuJoCo's binary STL loader accepts at most 200,000 faces per file.
        chunks = [
            triangles[start : start + 180000]
            for start in range(0, len(triangles), 180000)
        ]
        for index, chunk in enumerate(chunks):
            name = group if len(chunks) == 1 else f"{group}_{index}"
            path = output_dir / f"{name}.stl"
            count = write_binary_stl(path, np.asarray(chunk))
            components[name] = {"file": path.name, "triangles": count, "group": group}

    bounds = np.asarray(bounds)
    metadata = {
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cad_units": "millimetres",
        "mesh_units": "metres",
        "linear_deflection_mm": linear_deflection_mm,
        "angular_deflection_rad": angular_deflection_rad,
        "source_solids": solid_count,
        "bounds_m": [
            bounds[:, 0].min(axis=0).tolist(),
            bounds[:, 1].max(axis=0).tolist(),
        ],
        "components": components,
        "camera_reference": {
            "origin_kind": "CAD lens-opening centre; optical centre not calibrated to CAD",
            "position_cad_m": (np.asarray(centre) * 0.001).tolist(),
            "rotation_cad_from_opencv": camera_rotation.tolist(),
            "axes": "OpenCV: +X right, +Y down, +Z forward",
        },
    }
    (output_dir.parent / "geometry.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata
