"""Runtime FG-CLIP descriptors for finalized 3D instance hypotheses."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from vipe.utils.logging import pbar


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_FGCLIP_REVISION = "5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f"
FGCLIP_GRID = 24
CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture")


def _l2_numpy(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-8)


def _image_tensor(
    image,
    size: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    from PIL import Image

    pixels = torch.from_numpy(
        np.array(image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))
    ).float()
    pixels = pixels.permute(2, 0, 1) / 255.0
    pixels = pixels.unsqueeze(0).to(device)
    return (pixels - mean) / std


def _prompts(names: Sequence[str], template: str | None) -> list[str]:
    return [template.format(name) if template else name for name in names]


def _learned_temperature(module, default: float = 0.01) -> float:
    """Return the contrastive softmax temperature encoded by FG-CLIP."""
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        if name.endswith("logit_scale") and value.numel() == 1:
            return float(1.0 / value.exp().item())
    return default


class FGCLIPBackbone:
    """FG-CLIP dense features loaded from one explicitly pinned model revision."""

    patch_size = 14
    grid = FGCLIP_GRID
    dimension = 768

    def __init__(
        self,
        *,
        model_path: str | Path,
        revision: str,
        device: str | torch.device = "cuda",
    ) -> None:
        if not str(model_path) or not revision:
            raise ValueError("FG-CLIP requires explicit model_path and revision")
        if revision != _FGCLIP_REVISION:
            raise ValueError(
                f"FG-CLIP revision must be the audited {_FGCLIP_REVISION}, got {revision}"
            )

        import transformers

        if int(transformers.__version__.split(".")[0]) >= 5:
            raise RuntimeError(f"FG-CLIP requires transformers<5, found {transformers.__version__}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        load_args = {
            "pretrained_model_name_or_path": str(model_path),
            "revision": revision,
            "trust_remote_code": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(**load_args).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(**load_args)
        self.temperature = _learned_temperature(self.model)
        self.mean = torch.tensor(_CLIP_MEAN, device=self.device)[None, :, None, None]
        self.std = torch.tensor(_CLIP_STD, device=self.device)[None, :, None, None]

    @torch.no_grad()
    def dense_features(self, image) -> torch.Tensor:
        size = self.grid * self.patch_size
        pixels = _image_tensor(image, size, self.device, self.mean, self.std)
        dense = self.model.get_image_dense_features(pixels, interpolate_pos_encoding=True)
        dense = dense.reshape(self.grid, self.grid, -1).float()
        if dense.shape[-1] != self.dimension:
            raise RuntimeError(f"FG-CLIP returned D={dense.shape[-1]}, expected {self.dimension}")
        return dense

    @torch.no_grad()
    def encode_text(self, names: Sequence[str], template: str | None = None) -> torch.Tensor:
        tokens = self.tokenizer(
            _prompts(names, template),
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        features = self.model.get_text_features(tokens, walk_short_pos=True).float()
        if features.shape[-1] != self.dimension:
            raise RuntimeError(f"FG-CLIP returned text D={features.shape[-1]}, expected {self.dimension}")
        return features


def text_scores(
    backbone: FGCLIPBackbone,
    features: np.ndarray,
    prompts: Sequence[str],
    *,
    template: str | None = "a photo of a {}",
    negatives: Sequence[str] = CANONICAL_NEGATIVES,
    temperature: float | None = None,
) -> np.ndarray:
    """Return each query's softmax probability against the canonical negatives."""
    query_count = len(prompts)
    text = _l2_numpy(
        backbone.encode_text([*prompts, *negatives], template)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    similarities = _l2_numpy(np.asarray(features, dtype=np.float32)) @ text.T
    scaled = (similarities - similarities.max(axis=1, keepdims=True)) / float(
        backbone.temperature if temperature is None else temperature
    )
    exponentials = np.exp(scaled)
    negative_sum = exponentials[:, query_count:].sum(axis=1, keepdims=True)
    probabilities = exponentials[:, :query_count] / (
        exponentials[:, :query_count] + negative_sum
    )
    return probabilities.astype(np.float32)


class ProjectiveFeatureAccumulator:
    """Fuse dense image features directly onto fixed TSDF surface representatives."""

    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        dimension: int,
        *,
        device: str | torch.device = "cuda",
        weight_a: float = 1.0,
        weight_b: float = 1.0,
        occlusion_tolerance_m: float = 0.05,
    ) -> None:
        self.device = torch.device(device)
        self.points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        self.normals = torch.as_tensor(normals, dtype=torch.float32, device=self.device)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Expected points with shape (N, 3), got {tuple(self.points.shape)}")
        if self.normals.shape != self.points.shape:
            raise ValueError("Normals must match the point array shape")
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("Feature dimension must be positive")
        self.weight_a = float(weight_a)
        self.weight_b = float(weight_b)
        self.occlusion_tolerance_m = float(occlusion_tolerance_m)
        self.sum_wf = torch.zeros(
            len(self.points), self.dimension, dtype=torch.float32, device=self.device
        )
        self.sum_w = torch.zeros(len(self.points), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def integrate(
        self,
        feature_map: torch.Tensor,
        intrinsics: np.ndarray,
        c2w: np.ndarray,
        width: int,
        height: int,
        depth: np.ndarray,
    ) -> int:
        """Accumulate one frame using dense sensor-depth agreement for visibility."""
        feature_map = torch.as_tensor(feature_map, dtype=torch.float32, device=self.device)
        if (
            feature_map.ndim != 3
            or feature_map.shape[0] != feature_map.shape[1]
            or feature_map.shape[2] != self.dimension
        ):
            raise ValueError(f"Expected square (G, G, {self.dimension}) feature map")
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if depth.shape != (height, width):
            raise ValueError(f"Expected dense depth shape {(height, width)}, got {tuple(depth.shape)}")
        c2w = torch.as_tensor(c2w, dtype=torch.float32, device=self.device)
        if c2w.shape != (4, 4):
            raise ValueError(f"Expected c2w shape (4, 4), got {tuple(c2w.shape)}")
        fx, fy, cx, cy = (float(value) for value in np.asarray(intrinsics).reshape(4))

        camera_points = (self.points - c2w[:3, 3]) @ c2w[:3, :3]
        z = camera_points[:, 2]
        u = fx * camera_points[:, 0] / z + cx
        v = fy * camera_points[:, 1] / z + cy
        in_bounds = (
            (z > 1e-3)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        if not bool(in_bounds.any()):
            return 0

        ui = u.long().clamp(0, width - 1)
        vi = v.long().clamp(0, height - 1)
        sensor_depth = depth.reshape(-1)[vi * width + ui]
        visible = (
            in_bounds
            & (sensor_depth > 1e-3)
            & ((z - sensor_depth).abs() <= self.occlusion_tolerance_m)
        )
        indices = torch.where(visible)[0]
        if not indices.numel():
            return 0

        grid = feature_map.shape[0]
        cell_u = (u / width * grid).long().clamp(0, grid - 1)
        cell_v = (v / height * grid).long().clamp(0, grid - 1)
        cells = cell_v * grid + cell_u
        flat_features = feature_map.reshape(grid * grid, self.dimension)

        visible_z = z[indices]
        weights = (1.0 / visible_z.clamp_min(1e-3)) ** self.weight_b
        incidence_ray = camera_points[indices] / visible_z[:, None].clamp_min(1e-8)
        camera_normals = self.normals[indices] @ c2w[:3, :3]
        incidence = (camera_normals * incidence_ray).sum(-1).abs().clamp(0, 1)
        weights = (weights * incidence**self.weight_a).clamp_min(1e-6)

        self.sum_wf.index_add_(
            0,
            indices,
            flat_features[cells[indices]] * weights[:, None],
        )
        self.sum_w.index_add_(0, indices, weights)
        return int(indices.numel())

    @property
    def hit(self) -> torch.Tensor:
        return self.sum_w > 0

    @torch.no_grad()
    def point_features(self) -> np.ndarray:
        """Materialize the fused per-point field A and release no semantic information."""
        self.sum_wf.div_(self.sum_w.clamp_min(1e-8)[:, None])
        return self.sum_wf.cpu().numpy()


def distill_semantic_features(
    *,
    features: Mapping,
    points: np.ndarray,
    normals: np.ndarray,
    frame_indices: Sequence[int],
    rgb_of: Callable[[int], np.ndarray],
    depth_of: Callable[[int], np.ndarray],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, dict]:
    """Return the float32 per-point semantic field A and compact runtime metrics."""
    required = ("weight_a", "weight_b", "occlusion_tolerance_m", "model_path", "revision")
    try:
        values = {key: features[key] for key in required}
    except KeyError as error:
        raise ValueError(f"Missing semantic setting: {error.args[0]}") from error

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape (F, 4, 4), got {poses.shape}")
    selected_frames = sorted(int(index) for index in frame_indices)
    if selected_frames and (selected_frames[0] < 0 or selected_frames[-1] >= len(poses)):
        raise ValueError("Selected semantic frame index is out of range")

    backbone = FGCLIPBackbone(
        model_path=values["model_path"],
        revision=values["revision"],
        device=device,
    )
    accumulator = ProjectiveFeatureAccumulator(
        points,
        normals,
        backbone.dimension,
        device=device,
        weight_a=float(values["weight_a"]),
        weight_b=float(values["weight_b"]),
        occlusion_tolerance_m=float(values["occlusion_tolerance_m"]),
    )
    from PIL import Image

    for frame_index in pbar(selected_frames, desc="Semantic features", unit="frame"):
        rgb = rgb_of(frame_index)
        image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb))
        feature_map = backbone.dense_features(image)
        accumulator.integrate(
            feature_map,
            intrinsics,
            poses[frame_index],
            width,
            height,
            depth_of(frame_index),
        )

    hit_fraction = float(accumulator.hit.float().mean().item()) if len(points) else 0.0
    point_features = accumulator.point_features()
    metrics = {
        "grid": backbone.grid,
        "descriptor_dimension": backbone.dimension,
        "selected_frames": len(selected_frames),
        "valid_point_descriptor_count": int(np.count_nonzero(np.linalg.norm(point_features, axis=1))),
        "direct_point_hit_fraction": hit_fraction,
    }
    return point_features, metrics


@torch.no_grad()
def pool_hypothesis_descriptors(
    point_features: np.ndarray,
    hypotheses: Sequence[np.ndarray],
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 65536,
) -> np.ndarray:
    """Derive one float32 arithmetic-mean descriptor B per hypothesis from field A."""
    point_features = np.asarray(point_features, dtype=np.float32)
    if point_features.ndim != 2 or point_features.shape[1] == 0:
        raise ValueError("Expected point_features with shape (N, D)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    features = torch.as_tensor(point_features, dtype=torch.float32, device=device)
    observed = features.norm(dim=1) > 0
    descriptors = torch.zeros(
        len(hypotheses), features.shape[1], dtype=torch.float32, device=features.device
    )
    point_count = features.shape[0]
    for hypothesis_id, hypothesis in enumerate(hypotheses):
        hypothesis = np.asarray(hypothesis, dtype=np.int64)
        if hypothesis.ndim != 1 or ((hypothesis < 0) | (hypothesis >= point_count)).any():
            raise ValueError(f"Hypothesis {hypothesis_id} contains invalid point indices")
        feature_sum = torch.zeros(features.shape[1], dtype=torch.float32, device=features.device)
        observed_count = 0
        for start in range(0, len(hypothesis), chunk_size):
            indices = torch.as_tensor(
                hypothesis[start : start + chunk_size], dtype=torch.long, device=features.device
            )
            indices = indices[observed[indices]]
            if not indices.numel():
                continue
            feature_sum += features[indices].sum(0)
            observed_count += indices.numel()
        if observed_count:
            descriptors[hypothesis_id] = feature_sum / observed_count
    return descriptors.cpu().numpy()


class OverlapField:
    """Bounded reconstruction of the arithmetic-mean descriptor at each covered point."""

    def __init__(
        self,
        point_count: int,
        hypotheses: Sequence[np.ndarray],
        descriptors: np.ndarray,
    ) -> None:
        self.descriptors = np.asarray(descriptors, dtype=np.float32)
        if self.descriptors.ndim != 2 or self.descriptors.shape[0] != len(hypotheses):
            raise ValueError("Descriptors must have one row per hypothesis")

        valid = np.linalg.norm(self.descriptors, axis=1) > 0
        point_parts = []
        hypothesis_parts = []
        for hypothesis_id, hypothesis in enumerate(hypotheses):
            points = np.asarray(hypothesis, dtype=np.int64)
            if points.ndim != 1 or ((points < 0) | (points >= point_count)).any():
                raise ValueError(f"Hypothesis {hypothesis_id} contains invalid point indices")
            if not valid[hypothesis_id]:
                continue
            point_parts.append(points)
            hypothesis_parts.append(np.full(len(points), hypothesis_id, np.int32))

        if point_parts:
            points = np.concatenate(point_parts)
            members = np.concatenate(hypothesis_parts)
            order = np.argsort(points, kind="stable")
            points, self.members = points[order], members[order]
            counts = np.bincount(points, minlength=point_count)
        else:
            self.members = np.empty(0, np.int32)
            counts = np.zeros(point_count, np.int64)
        self.offsets = np.empty(point_count + 1, np.int64)
        self.offsets[0] = 0
        np.cumsum(counts, out=self.offsets[1:])
        self.covered = counts > 0

    def rows(self, point_indices: np.ndarray) -> np.ndarray:
        """Reconstruct selected rows without allocating the complete point-feature field."""
        point_indices = np.asarray(point_indices, dtype=np.int64)
        if point_indices.ndim != 1 or (
            (point_indices < 0) | (point_indices >= len(self.covered))
        ).any():
            raise ValueError("Point indices are out of range")
        counts = self.offsets[point_indices + 1] - self.offsets[point_indices]
        result = np.zeros((len(point_indices), self.descriptors.shape[1]), np.float32)
        total = int(counts.sum())
        if not total:
            return result

        rows = np.repeat(np.arange(len(point_indices)), counts)
        starts = np.repeat(self.offsets[point_indices], counts)
        local = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        descriptor_ids = self.members[starts + local]
        np.add.at(result, rows, self.descriptors[descriptor_ids])
        observed = counts > 0
        result[observed] /= counts[observed, None]
        return result

    def chunks(self, chunk_size: int = 65536) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, len(self.covered), chunk_size):
            stop = min(start + chunk_size, len(self.covered))
            covered = self.covered[start:stop]
            values = np.zeros((stop - start, self.descriptors.shape[1]), np.float32)
            if covered.any():
                local = np.flatnonzero(covered)
                values[local] = self.rows(local + start)
            yield slice(start, stop), values, covered


def _write_semantic_pca_ply(
    path: str | Path,
    points: np.ndarray,
    normals: np.ndarray,
    covered: np.ndarray,
    dimension: int,
    rows_of: Callable[[np.ndarray], np.ndarray],
    chunks_of: Callable[[], Iterator[tuple[slice, np.ndarray, np.ndarray]]],
    *,
    max_pca_samples: int = 100000,
) -> float:
    """Write one deterministic PCA-colored point field."""
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("Points and normals must have matching (N, 3) shapes")
    covered = np.asarray(covered, dtype=bool)
    if covered.shape != (len(points),):
        raise ValueError("Semantic coverage must have one value per point")
    if max_pca_samples <= 0:
        raise ValueError("max_pca_samples must be positive")

    covered_indices = np.flatnonzero(covered)
    covered_count = len(covered_indices)
    mean = np.zeros(dimension, np.float64)
    for _, values, chunk_covered in chunks_of():
        mean += values[chunk_covered].sum(0, dtype=np.float64)
    if covered_count:
        mean = (mean / covered_count).astype(np.float32)
        sample_indices = np.random.default_rng(0).choice(
            covered_indices, min(covered_count, max_pca_samples), replace=False
        )
        sample = rows_of(sample_indices)
        _, _, right = np.linalg.svd(sample - mean, full_matrices=False)
        components = np.zeros((3, dimension), np.float32)
        components[: min(3, len(right))] = right[:3]
        projected_sample = (sample - mean) @ components.T
        low = np.percentile(projected_sample, 2, axis=0)
        high = np.percentile(projected_sample, 98, axis=0)
    else:
        components = np.zeros((3, field.descriptors.shape[1]), np.float32)
        low, high = np.zeros(3), np.ones(3)

    record_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        for point_slice, values, chunk_covered in chunks_of():
            records = np.empty(point_slice.stop - point_slice.start, record_dtype)
            records["x"], records["y"], records["z"] = points[point_slice].T
            records["nx"], records["ny"], records["nz"] = normals[point_slice].T
            colors = np.full((len(records), 3), 128, np.uint8)
            if chunk_covered.any():
                projection = (values[chunk_covered] - mean) @ components.T
                colors[chunk_covered] = (
                    np.clip((projection - low) / (high - low + 1e-8), 0, 1) * 255
                ).astype(np.uint8)
            records["red"], records["green"], records["blue"] = colors.T
            handle.write(records.tobytes())
    return covered_count / len(points) if len(points) else 0.0


def write_point_semantic_pca_ply(
    path: str | Path,
    points: np.ndarray,
    normals: np.ndarray,
    point_features: np.ndarray,
    *,
    chunk_size: int = 65536,
    max_pca_samples: int = 100000,
) -> float:
    """Write PCA colors for the directly fused per-point field A."""
    point_features = np.asarray(point_features, dtype=np.float32)
    if point_features.ndim != 2 or point_features.shape[0] != len(points):
        raise ValueError("Point features must have shape (N, D)")
    covered = np.linalg.norm(point_features, axis=1) > 0

    def chunks():
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            point_slice = slice(start, stop)
            yield point_slice, point_features[point_slice], covered[point_slice]

    return _write_semantic_pca_ply(
        path,
        points,
        normals,
        covered,
        point_features.shape[1],
        lambda indices: point_features[indices],
        chunks,
        max_pca_samples=max_pca_samples,
    )


def write_hypothesis_average_semantic_pca_ply(
    path: str | Path,
    points: np.ndarray,
    normals: np.ndarray,
    point_features: np.ndarray,
    hypotheses: Sequence[np.ndarray],
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 65536,
    max_pca_samples: int = 100000,
) -> tuple[float, int]:
    """Derive B and write PCA colors for the hypothesis-averaged point field C."""
    descriptors = pool_hypothesis_descriptors(point_features, hypotheses, device=device)
    field = OverlapField(len(points), hypotheses, descriptors)
    coverage = _write_semantic_pca_ply(
        path,
        points,
        normals,
        field.covered,
        descriptors.shape[1],
        field.rows,
        lambda: field.chunks(chunk_size),
        max_pca_samples=max_pca_samples,
    )
    valid_count = int(np.count_nonzero(np.linalg.norm(descriptors, axis=1)))
    return coverage, valid_count
