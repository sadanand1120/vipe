from __future__ import annotations

import re

from pathlib import Path

import numpy as np
import open3d as o3d

from vipe.bench.scannet import AttrDict, ScanNetDataset, ScanNetEvaluator


STANDARD_REPLICA_SCENES = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)


def full_replica_scene_candidates(scene_name: str) -> list[str]:
    candidates = [scene_name]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", scene_name)
    if match is not None:
        candidates.append(f"{match.group(1)}_{match.group(2)}")
    return candidates


PLY_DTYPES = {
    "char": np.dtype("i1"),
    "int8": np.dtype("i1"),
    "uchar": np.dtype("u1"),
    "uint8": np.dtype("u1"),
    "short": np.dtype("<i2"),
    "int16": np.dtype("<i2"),
    "ushort": np.dtype("<u2"),
    "uint16": np.dtype("<u2"),
    "int": np.dtype("<i4"),
    "int32": np.dtype("<i4"),
    "uint": np.dtype("<u4"),
    "uint32": np.dtype("<u4"),
    "float": np.dtype("<f4"),
    "float32": np.dtype("<f4"),
    "double": np.dtype("<f8"),
    "float64": np.dtype("<f8"),
}


def _parse_binary_ply_header(handle) -> tuple[int, int, list[tuple[str, np.dtype]], tuple[np.dtype, np.dtype]]:
    header = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading Replica PLY header")
        text = line.decode("ascii").strip()
        header.append(text)
        if text == "end_header":
            break

    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Replica GT mesh loader expects binary_little_endian PLY")

    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, np.dtype]] = []
    face_list_type: tuple[np.dtype, np.dtype] | None = None
    current_element = None

    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            elif current_element == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3:
                raise ValueError(f"Unsupported Replica vertex property: {line}")
            vertex_properties.append((parts[2], PLY_DTYPES[parts[1]]))
        elif parts[:2] == ["property", "list"] and current_element == "face":
            face_list_type = (PLY_DTYPES[parts[2]], PLY_DTYPES[parts[3]])

    if vertex_count <= 0 or face_count <= 0 or face_list_type is None:
        raise ValueError("Replica GT mesh PLY is missing vertex/face data")
    return vertex_count, face_count, vertex_properties, face_list_type


def _triangulate_faces(handle, face_count: int, count_dtype: np.dtype, index_dtype: np.dtype) -> np.ndarray:
    # Replica_full mesh.ply stores every face as a quad: uint8 count + 4 int32 vertex indices.
    if count_dtype.itemsize == 1 and index_dtype.itemsize == 4:
        start = handle.tell()
        fixed_quad_dtype = np.dtype([("count", count_dtype), ("indices", index_dtype, (4,))])
        faces = np.fromfile(handle, dtype=fixed_quad_dtype, count=face_count)
        if len(faces) == face_count and bool(np.all(faces["count"] == 4)):
            quads = faces["indices"].astype(np.int32, copy=False)
            triangles = np.empty((face_count * 2, 3), dtype=np.int32)
            triangles[0::2] = quads[:, [0, 1, 2]]
            triangles[1::2] = quads[:, [0, 2, 3]]
            return triangles
        handle.seek(start)

    triangles = []
    for _ in range(face_count):
        count = int(np.fromfile(handle, dtype=count_dtype, count=1)[0])
        indices = np.fromfile(handle, dtype=index_dtype, count=count).astype(np.int32, copy=False)
        if count < 3:
            continue
        for i in range(1, count - 1):
            triangles.append((indices[0], indices[i], indices[i + 1]))
    return np.asarray(triangles, dtype=np.int32)


def load_replica_quad_mesh(mesh_path: str | Path) -> o3d.geometry.TriangleMesh:
    mesh_path = Path(mesh_path)
    with mesh_path.open("rb") as handle:
        vertex_count, face_count, vertex_properties, face_list_type = _parse_binary_ply_header(handle)
        vertex_dtype = np.dtype([(name, dtype) for name, dtype in vertex_properties])
        vertices = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        if len(vertices) != vertex_count:
            raise ValueError(f"Replica GT mesh vertex data is truncated: {mesh_path}")
        triangles = _triangulate_faces(handle, face_count, *face_list_type)

    for name in ("x", "y", "z"):
        if name not in vertices.dtype.names:
            raise ValueError(f"Replica GT mesh is missing vertex property '{name}': {mesh_path}")
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64, copy=False)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    if all(name in vertices.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.float64) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if all(name in vertices.dtype.names for name in ("nx", "ny", "nz")):
        normals = np.column_stack([vertices["nx"], vertices["ny"], vertices["nz"]]).astype(np.float64, copy=False)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return mesh


class ReplicaDataset(ScanNetDataset):
    DATASET_LABEL = "Replica"

    def _load_scene_list(self, input_root: Path) -> list[str]:
        if not input_root.exists():
            return []
        available = {path.name for path in input_root.iterdir() if path.is_dir()}
        ordered = [scene for scene in STANDARD_REPLICA_SCENES if scene in available]
        extras = sorted(available - set(ordered))
        return ordered + extras

    def _gt_mesh_path(self, scene: str) -> Path:
        checked = []
        for candidate in full_replica_scene_candidates(scene):
            mesh_path = self.raw_root / candidate / "mesh.ply"
            checked.append(str(mesh_path))
            if mesh_path.is_file():
                return mesh_path
        raise FileNotFoundError(f"Missing {self.DATASET_LABEL} GT mesh for {scene}. Checked: {checked}")

    def _load_gt_mesh(self, mesh_path: str | Path) -> o3d.geometry.TriangleMesh:
        return load_replica_quad_mesh(mesh_path)


class ReplicaEvaluator(ScanNetEvaluator):
    DATASET_KEY = "replica"
    DATASET_LABEL = "Replica"

    def _build_datasets(self, input_root: Path, raw_root: Path) -> AttrDict:
        return AttrDict(replica=ReplicaDataset(input_root, raw_root, self.config))
