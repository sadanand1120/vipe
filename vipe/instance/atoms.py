"""Normal-aware atomization of the instance occupancy cloud."""

import numpy as np


def estimate_normals(points: np.ndarray, radius: float, max_neighbors: int) -> np.ndarray:
    """Estimate surface normals with the frontier's Open3D implementation."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbors)
    )
    return np.asarray(cloud.normals, np.float32)


def _knn_graph(points: np.ndarray, k: int, radius_cap: float):
    """Return symmetric, deduplicated kNN edges as ``(lo, hi, distance)``."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distance, neighbors = tree.query(points, k=k + 1)
    rows = np.repeat(np.arange(len(points)), k)
    cols = neighbors[:, 1:].ravel()
    distance = distance[:, 1:].ravel()
    keep = distance <= radius_cap
    rows, cols, distance = rows[keep], cols[keep], distance[keep]
    lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
    key = lo.astype(np.int64) * len(points) + hi
    _, unique = np.unique(key, return_index=True)
    return lo[unique], hi[unique], distance[unique]


def _seeds(points: np.ndarray, seed_resolution: float) -> np.ndarray:
    """Select the point nearest each occupied seed-cell centroid."""
    cells = np.floor(points / seed_resolution).astype(np.int64)
    key = cells[:, 0] * 73856093 ^ cells[:, 1] * 19349663 ^ cells[:, 2] * 83492791
    _, inverse = np.unique(key, return_inverse=True)
    count = inverse.max() + 1
    centroids = np.zeros((count, 3))
    sizes = np.zeros(count)
    np.add.at(centroids, inverse, points)
    np.add.at(sizes, inverse, 1.0)
    centroids /= sizes[:, None]
    distance2 = ((points - centroids[inverse]) ** 2).sum(1)
    seeds = np.full(count, -1, np.int64)
    for index in np.argsort(distance2):
        cell = inverse[index]
        if seeds[cell] < 0:
            seeds[cell] = index
    return seeds[seeds >= 0]


def build_atoms(
    points: np.ndarray,
    normals: np.ndarray,
    seed_resolution: float,
    k: int,
    normal_weight: float,
    radius_cap_voxels: float,
    voxel_size: float,
):
    """Partition points by multi-source Dijkstra on a normal-weighted kNN graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    count = len(points)
    lo, hi, distance = _knn_graph(
        points, k=k, radius_cap=radius_cap_voxels * voxel_size * np.sqrt(3)
    )
    normal_agreement = (normals[lo] * normals[hi]).sum(1)
    cost = distance * (1.0 + normal_weight * (1.0 - np.abs(normal_agreement)))
    graph = csr_matrix(
        (
            np.concatenate([cost, cost]),
            (np.concatenate([lo, hi]), np.concatenate([hi, lo])),
        ),
        shape=(count, count),
    )
    seeds = _seeds(points, seed_resolution)
    _, _, source = dijkstra(
        graph, directed=False, indices=seeds, min_only=True, return_predecessors=True
    )
    unique, atom_of = np.unique(source, return_inverse=True)
    atom_of = atom_of.astype(np.int64)
    if unique[0] < 0:
        unreachable = atom_of == 0
        atom_of[unreachable] = atom_of.max() + 1 + np.arange(unreachable.sum())
        _, atom_of = np.unique(atom_of, return_inverse=True)
    return atom_of, (lo, hi, distance)


def atom_adjacency(atom_of: np.ndarray, edges):
    """Build atom adjacency and cross-atom voxel-edge counts."""
    lo, hi, _ = edges
    atom_lo, atom_hi = atom_of[lo], atom_of[hi]
    keep = atom_lo != atom_hi
    left = np.minimum(atom_lo[keep], atom_hi[keep])
    right = np.maximum(atom_lo[keep], atom_hi[keep])
    atom_count = atom_of.max() + 1
    key = left * atom_count + right
    unique, count = np.unique(key, return_counts=True)
    return (
        (unique // atom_count).astype(np.int64),
        (unique % atom_count).astype(np.int64),
        count.astype(np.int64),
    )


def atom_voxels(atom_of: np.ndarray) -> list[np.ndarray]:
    """Return voxel indices grouped by ascending atom ID."""
    order = np.argsort(atom_of, kind="stable").astype(np.int32)
    sorted_atoms = atom_of[order]
    bounds = np.searchsorted(sorted_atoms, np.arange(sorted_atoms.max() + 2))
    return [order[bounds[i] : bounds[i + 1]] for i in range(sorted_atoms.max() + 1)]
