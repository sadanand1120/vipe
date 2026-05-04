import argparse
import os
import sys

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


ORIENT_FILE = "orient.npz"
ORIENT_FORMAT = "vipe_orient_v1"
ORIENTATION_AXIS_LENGTH = 0.4
MIN_RENDER_AXIS_LENGTH = 1e-3
DEFAULT_ROTATION_STEP_DEG = 5.0
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 960
DEFAULT_POINT_SIZE = 1.5
DEFAULT_HIGHLIGHT_POINT_SIZE = 4.0
HIGHLIGHT_COLOR = np.array([1.0, 0.82, 0.10], dtype=np.float32)
BASE_CLOUD_NAME = "rgb"
HIGHLIGHT_CLOUD_NAME = "current_instance"
CURRENT_AXES_NAME = "current_orientation_axes"
ALL_AXES_NAME = "all_orientation_axes"

INSTANCE_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("instance_id", "<i4"),
    ]
)


@dataclass(frozen=True)
class OrientPaths:
    pcd_dir: Path
    rgb_path: Path
    instance_path: Path
    orient_path: Path


def resolve_paths(input_path: str) -> OrientPaths:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        pcd_dir = path.parent
        rgb_path = path
    elif (path / "pcd" / "rgb.ply").exists():
        pcd_dir = path / "pcd"
        rgb_path = pcd_dir / "rgb.ply"
    elif (path / "rgb.ply").exists():
        pcd_dir = path
        rgb_path = pcd_dir / "rgb.ply"
    else:
        raise FileNotFoundError(f"Could not find rgb.ply from input path: {path}")

    instance_path = pcd_dir / "instance.ply"
    if not instance_path.exists():
        raise FileNotFoundError(f"Missing instance PLY: {instance_path}")
    return OrientPaths(pcd_dir, rgb_path, instance_path, pcd_dir / ORIENT_FILE)


def read_binary_ply_header(path: Path) -> tuple[int, list[str], int]:
    vertex_count = None
    properties = []
    with path.open("rb") as f:
        first = f.readline().decode("ascii").strip()
        fmt = f.readline().decode("ascii").strip()
        if first != "ply" or fmt != "format binary_little_endian 1.0":
            raise ValueError(f"{path} must be a binary_little_endian PLY")

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Missing end_header in {path}")
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property "):
                properties.append(line)
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {path}")
        return vertex_count, properties, f.tell()


def read_instance_ids(path: Path) -> np.ndarray:
    expected_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property int instance_id",
    ]
    vertex_count, properties, body_offset = read_binary_ply_header(path)
    if properties != expected_properties:
        raise ValueError(f"Unexpected instance PLY schema in {path}: {properties}")

    with path.open("rb") as f:
        f.seek(body_offset)
        vertices = np.fromfile(f, dtype=INSTANCE_VERTEX_DTYPE, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(f"Instance PLY ended early: expected {vertex_count}, found {len(vertices)}")
    return vertices["instance_id"].astype(np.int32, copy=False)


def filter_instance_ids(labels: np.ndarray, min_instance_points: int) -> tuple[np.ndarray, np.ndarray]:
    valid = labels >= 0
    if not valid.any():
        raise ValueError("No non-negative instance IDs found in instance.ply")

    instance_ids, point_counts = np.unique(labels[valid], return_counts=True)
    keep = point_counts >= int(min_instance_points)
    instance_ids = instance_ids[keep].astype(np.int32, copy=False)
    point_counts = point_counts[keep].astype(np.int32, copy=False)
    if len(instance_ids) == 0:
        raise ValueError("No instances remain after min-instance-points filtering")
    return instance_ids, point_counts


def compute_centroids(points: np.ndarray, labels: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    valid_indices = np.flatnonzero(labels >= 0)
    positions = np.searchsorted(instance_ids, labels[valid_indices])
    matched = positions < len(instance_ids)
    matched[matched] = instance_ids[positions[matched]] == labels[valid_indices][matched]
    positions = positions[matched]
    point_indices = valid_indices[matched]

    counts = np.bincount(positions, minlength=len(instance_ids)).astype(np.float32)
    sums = np.empty((len(instance_ids), 3), dtype=np.float32)
    for dim in range(3):
        sums[:, dim] = np.bincount(positions, weights=points[point_indices, dim], minlength=len(instance_ids))
    return sums / counts[:, None]


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float32), full_matrices=False)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation.astype(np.float32, copy=False)


def rotation_to_axis_vectors(rotation: np.ndarray) -> np.ndarray:
    return (orthonormalize_rotation(rotation) * ORIENTATION_AXIS_LENGTH).T.astype(np.float32, copy=False)


def axis_vectors_to_rotation(axis_vectors: np.ndarray) -> np.ndarray:
    axis_vectors = np.asarray(axis_vectors, dtype=np.float32)
    lengths = np.linalg.norm(axis_vectors, axis=1)
    scale = float(lengths.mean()) if len(lengths) else 0.0
    if scale <= 0.0:
        return np.eye(3, dtype=np.float32)
    return orthonormalize_rotation((axis_vectors / scale).T)


def rotation_matrix_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def tint_colors(colors: np.ndarray, tint: np.ndarray, amount: float = 0.65) -> np.ndarray:
    return np.clip(colors * (1.0 - amount) + tint.reshape(1, 3) * amount, 0.0, 1.0)


def build_axes_mesh(centroid: np.ndarray, rotation: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        orthonormalize_rotation(rotation) * max(ORIENTATION_AXIS_LENGTH, MIN_RENDER_AXIS_LENGTH)
    ).astype(np.float64, copy=False)
    transform[:3, 3] = centroid.astype(np.float64, copy=False)
    mesh.transform(transform)
    return mesh


def dump_orientations(
    path: Path,
    rgb_path: Path,
    instance_path: Path,
    min_instance_points: int,
    instance_ids: np.ndarray,
    centroids: np.ndarray,
    rotations: np.ndarray,
    point_counts: np.ndarray,
) -> None:
    axes = np.stack([rotation_to_axis_vectors(rotation) for rotation in rotations], axis=0)
    np.savez_compressed(
        path,
        format=np.asarray(ORIENT_FORMAT),
        input_ply=np.asarray(rgb_path.name),
        instance_ply=np.asarray(instance_path.name),
        min_instance_points=np.asarray(int(min_instance_points), dtype=np.int32),
        axis_length=np.asarray(ORIENTATION_AXIS_LENGTH, dtype=np.float32),
        instance_ids=instance_ids.astype(np.int32, copy=False),
        point_counts=point_counts.astype(np.int32, copy=False),
        centroids=centroids.astype(np.float32, copy=False),
        rotations=rotations.astype(np.float32, copy=False),
        axes=axes.astype(np.float32, copy=False),
    )


def load_complete_orientations(path: Path, expected_instance_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing orientation annotation file: {path}")

    data = np.load(path)
    if data["format"].item() != ORIENT_FORMAT:
        raise ValueError(f"Unsupported orientation format in {path}: {data['format'].item()}")

    instance_ids = data["instance_ids"].astype(np.int32, copy=False)
    missing = np.setdiff1d(expected_instance_ids, instance_ids)
    extra = np.setdiff1d(instance_ids, expected_instance_ids)
    if len(instance_ids) != len(expected_instance_ids) or len(missing) or len(extra):
        raise ValueError(
            f"Incomplete orientation annotations in {path}: "
            f"expected={len(expected_instance_ids)} found={len(instance_ids)} "
            f"missing={missing.tolist()} extra={extra.tolist()}"
        )

    axes = data["axes"].astype(np.float32, copy=False)
    if axes.shape != (len(instance_ids), 3, 3):
        raise ValueError(f"Bad axes shape in {path}: {axes.shape}")
    centroids = data["centroids"].astype(np.float32, copy=False)
    if centroids.shape != (len(instance_ids), 3):
        raise ValueError(f"Bad centroids shape in {path}: {centroids.shape}")

    order = np.argsort(instance_ids)
    centroids = centroids[order]
    axes = axes[order]
    rotations = np.stack([axis_vectors_to_rotation(axis) for axis in axes], axis=0)
    return centroids, rotations


class OrientationAnnotator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.paths = resolve_paths(args.input_path)
        self.viz_mode = bool(args.viz)
        self.finished = False
        self.current_label3d = None

        self.pcd = o3d.io.read_point_cloud(str(self.paths.rgb_path))
        self.points = np.asarray(self.pcd.points, dtype=np.float32)
        self.colors = np.asarray(self.pcd.colors, dtype=np.float32)
        if len(self.points) == 0:
            raise ValueError(f"No points found in {self.paths.rgb_path}")
        if self.colors.shape[0] != self.points.shape[0]:
            raise ValueError(f"RGB point count mismatch in {self.paths.rgb_path}")

        self.labels = read_instance_ids(self.paths.instance_path)
        if self.labels.shape[0] != self.points.shape[0]:
            raise ValueError(
                f"Instance point count mismatch: rgb={self.points.shape[0]} instance={self.labels.shape[0]}"
            )

        self.instance_ids, self.point_counts = filter_instance_ids(self.labels, args.min_instance_points)
        self.centroids = compute_centroids(self.points, self.labels, self.instance_ids)
        self.rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(self.instance_ids), axis=0)
        self.rotation_initialized = np.zeros(len(self.instance_ids), dtype=bool)
        self.rotation_initialized[0] = True
        if self.viz_mode:
            self.centroids, self.rotations = load_complete_orientations(self.paths.orient_path, self.instance_ids)
            self.rotation_initialized[:] = True

        self.current_index = 0
        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("ViPE Instance Orientation Annotator", args.window_width, args.window_height)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self.window.set_on_close(self._on_close)

        self.info_label = gui.Label("")
        help_text = (
            "Read-only viz: B/N inspect instances, close exits"
            if self.viz_mode
            else "Keys: Up/Down pitch  Left/Right yaw  Q/E roll  R reset  B previous  N next/save"
        )
        self.help_label = gui.Label(help_text)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.show_skybox(False)

        self.window.add_child(self.info_label)
        self.window.add_child(self.help_label)
        self.window.add_child(self.scene_widget)

        base_material = rendering.MaterialRecord()
        base_material.shader = "defaultUnlit"
        base_material.point_size = float(args.point_size)
        self.scene_widget.scene.add_geometry(BASE_CLOUD_NAME, self.pcd, base_material)

        self.highlight_material = rendering.MaterialRecord()
        self.highlight_material.shader = "defaultUnlit"
        self.highlight_material.point_size = float(args.highlight_point_size)

        self.axes_material = rendering.MaterialRecord()
        self.axes_material.shader = "defaultUnlit"
        if self.viz_mode:
            all_axes = o3d.geometry.TriangleMesh()
            for centroid, rotation in zip(self.centroids, self.rotations, strict=True):
                all_axes += build_axes_mesh(centroid, rotation)
            self.scene_widget.scene.add_geometry(ALL_AXES_NAME, all_axes, self.axes_material)
        else:
            self.axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0])
            self.scene_widget.scene.add_geometry(CURRENT_AXES_NAME, self.axis_mesh, self.axes_material)

        bbox = self.pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())
        self._update_current_instance()

    def _on_layout(self, layout_context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        em = int(np.ceil(layout_context.theme.font_size))
        margin = max(8, em // 2)
        info_height = em * 4
        help_height = em * 2
        self.info_label.frame = gui.Rect(rect.x + margin, rect.y + margin, rect.width - 2 * margin, info_height)
        self.help_label.frame = gui.Rect(
            rect.x + margin,
            rect.y + margin + info_height,
            rect.width - 2 * margin,
            help_height,
        )
        scene_y = rect.y + margin + info_height + help_height + margin
        self.scene_widget.frame = gui.Rect(rect.x, scene_y, rect.width, max(1, rect.height - (scene_y - rect.y)))

    def _current_axis_vectors(self) -> np.ndarray:
        return rotation_to_axis_vectors(self.rotations[self.current_index])

    def _update_info_label(self) -> None:
        instance_id = int(self.instance_ids[self.current_index])
        centroid = self.centroids[self.current_index]
        axes = self._current_axis_vectors()
        mode = "viz" if self.viz_mode else "annotate"
        self.info_label.text = (
            f"Mode={mode}  Instance {self.current_index + 1}/{len(self.instance_ids)}"
            f"  id={instance_id}  points={int(self.point_counts[self.current_index])}"
            f"  axis_len={ORIENTATION_AXIS_LENGTH:.3f}\n"
            f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})\n"
            f"x={axes[0, 0]: .3f} {axes[0, 1]: .3f} {axes[0, 2]: .3f}    "
            f"y={axes[1, 0]: .3f} {axes[1, 1]: .3f} {axes[1, 2]: .3f}    "
            f"z={axes[2, 0]: .3f} {axes[2, 1]: .3f} {axes[2, 2]: .3f}"
        )

    def _update_highlight_geometry(self) -> None:
        mask = self.labels == int(self.instance_ids[self.current_index])
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(self.points[mask].astype(np.float64, copy=False))
        instance_pcd.colors = o3d.utility.Vector3dVector(
            tint_colors(self.colors[mask], HIGHLIGHT_COLOR).astype(np.float64, copy=False)
        )
        if self.scene_widget.scene.has_geometry(HIGHLIGHT_CLOUD_NAME):
            self.scene_widget.scene.remove_geometry(HIGHLIGHT_CLOUD_NAME)
        self.scene_widget.scene.add_geometry(HIGHLIGHT_CLOUD_NAME, instance_pcd, self.highlight_material)

    def _update_axes_geometry(self) -> None:
        if self.viz_mode:
            return

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = (
            self.rotations[self.current_index] * max(ORIENTATION_AXIS_LENGTH, MIN_RENDER_AXIS_LENGTH)
        ).astype(np.float64, copy=False)
        transform[:3, 3] = self.centroids[self.current_index].astype(np.float64, copy=False)
        self.scene_widget.scene.set_geometry_transform(CURRENT_AXES_NAME, transform)

    def _update_current_label(self) -> None:
        if self.current_label3d is not None:
            self.scene_widget.remove_3d_label(self.current_label3d)
        self.current_label3d = self.scene_widget.add_3d_label(
            self.centroids[self.current_index].astype(np.float32, copy=False),
            f"{self.current_index + 1}/{len(self.instance_ids)}",
        )

    def _update_current_instance(self) -> None:
        self._update_info_label()
        self._update_highlight_geometry()
        self._update_axes_geometry()
        self._update_current_label()
        self.window.post_redraw()

    def _rotate_current(self, local_rotation: np.ndarray) -> None:
        self.rotations[self.current_index] = orthonormalize_rotation(self.rotations[self.current_index] @ local_rotation)
        self._update_info_label()
        self._update_axes_geometry()
        self.window.post_redraw()

    def _advance(self, delta: int) -> None:
        new_index = int(np.clip(self.current_index + delta, 0, len(self.instance_ids) - 1))
        if new_index != self.current_index:
            if delta > 0 and not self.rotation_initialized[new_index]:
                self.rotations[new_index] = self.rotations[self.current_index].copy()
                self.rotation_initialized[new_index] = True
            self.current_index = new_index
            self._update_current_instance()

    def _save_and_close(self) -> None:
        dump_orientations(
            self.paths.orient_path,
            self.paths.rgb_path,
            self.paths.instance_path,
            self.args.min_instance_points,
            self.instance_ids,
            self.centroids,
            self.rotations,
            self.point_counts,
        )
        self.finished = True
        print(self.paths.orient_path)
        self.window.close()

    def _on_key(self, event: gui.KeyEvent) -> bool:
        if event.type != gui.KeyEvent.DOWN:
            return False

        if event.key == gui.KeyName.B:
            self._advance(-1)
            return True
        if event.key == gui.KeyName.N:
            if self.viz_mode:
                self._advance(1)
            elif self.current_index == len(self.instance_ids) - 1:
                self._save_and_close()
            else:
                self._advance(1)
            return True
        if self.viz_mode:
            return False

        step_rad = np.deg2rad(float(self.args.rotation_step_deg))
        if event.key == gui.KeyName.UP:
            self._rotate_current(rotation_matrix_x(step_rad))
            return True
        if event.key == gui.KeyName.DOWN:
            self._rotate_current(rotation_matrix_x(-step_rad))
            return True
        if event.key == gui.KeyName.LEFT:
            self._rotate_current(rotation_matrix_y(step_rad))
            return True
        if event.key == gui.KeyName.RIGHT:
            self._rotate_current(rotation_matrix_y(-step_rad))
            return True
        if event.key == gui.KeyName.Q:
            self._rotate_current(rotation_matrix_z(step_rad))
            return True
        if event.key == gui.KeyName.E:
            self._rotate_current(rotation_matrix_z(-step_rad))
            return True
        if event.key == gui.KeyName.R:
            self.rotations[self.current_index] = np.eye(3, dtype=np.float32)
            self._update_current_instance()
            return True
        return False

    def _on_close(self) -> bool:
        if not self.viz_mode and not self.finished:
            print("Closed without saving orientations.")
        gui.Application.instance.quit()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively annotate per-instance orientation axes on a ViPE output pointcloud."
    )
    parser.add_argument("input_path", help="Path to ViPE output dir, pcd dir, or rgb.ply.")
    parser.add_argument("--viz", action="store_true", help="Open read-only visualization of an existing complete orient.npz.")
    parser.add_argument("--min-instance-points", type=int, default=1)
    parser.add_argument("--rotation-step-deg", type=float, default=DEFAULT_ROTATION_STEP_DEG)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--highlight-point-size", type=float, default=DEFAULT_HIGHLIGHT_POINT_SIZE)
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH)
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT)
    return parser.parse_args()


def main() -> None:
    annotator = OrientationAnnotator(parse_args())
    gui.Application.instance.run()
    if annotator.finished:
        print(
            {
                "orientation_path": str(annotator.paths.orient_path),
                "n_instances": int(len(annotator.instance_ids)),
                "min_instance_points": int(annotator.args.min_instance_points),
            }
        )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
