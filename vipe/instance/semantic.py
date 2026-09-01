"""Runtime FG-CLIP descriptors for finalized 3D instance hypotheses."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from vipe.utils.logging import pbar


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_FGCLIP_REVISION = "5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f"
CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture")


def _l2_torch(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _l2_numpy(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-8)


def _image_tensor(image, size: int, device: torch.device) -> torch.Tensor:
    from PIL import Image

    pixels = torch.from_numpy(
        np.array(image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))
    ).float()
    pixels = pixels.permute(2, 0, 1) / 255.0
    pixels = (pixels - torch.tensor(_CLIP_MEAN)[:, None, None]) / torch.tensor(_CLIP_STD)[:, None, None]
    return pixels.unsqueeze(0).to(device)


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
    dimension = 768

    def __init__(
        self,
        *,
        grid: int,
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

        self.grid = int(grid)
        self.device = torch.device(device)
        load_args = {
            "pretrained_model_name_or_path": str(model_path),
            "revision": revision,
            "trust_remote_code": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(**load_args).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(**load_args)
        self.temperature = _learned_temperature(self.model)

    @torch.no_grad()
    def dense_features(self, image) -> torch.Tensor:
        pixels = _image_tensor(image, self.grid * self.patch_size, self.device)
        dense = self.model.get_image_dense_features(pixels, interpolate_pos_encoding=True)
        dense = dense.reshape(self.grid, self.grid, -1).float()
        if dense.shape[-1] != self.dimension:
            raise RuntimeError(f"FG-CLIP returned D={dense.shape[-1]}, expected {self.dimension}")
        return _l2_torch(dense)

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
        return _l2_torch(features)


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
    text = (
        backbone.encode_text([*prompts, *negatives], template)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    similarities = np.asarray(features, dtype=np.float32) @ text.T
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

    def pool_descriptors(
        self,
        hypotheses: Sequence[np.ndarray],
        *,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        return pool_instance_descriptors(
            self.sum_wf,
            self.sum_w,
            hypotheses,
            chunk_size=chunk_size,
        )


def distill_semantic_features(
    *,
    features: Mapping,
    points: np.ndarray,
    normals: np.ndarray,
    hypotheses: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    rgb_of: Callable[[int], np.ndarray],
    depth_of: Callable[[int], np.ndarray],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, dict]:
    """Return row-aligned float16 hypothesis descriptors and compact runtime metrics."""
    required = ("grid", "weight_a", "weight_b", "occlusion_tolerance_m", "model_path", "revision")
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
        grid=int(values["grid"]),
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

    instance_features = accumulator.pool_descriptors(hypotheses)
    hit_fraction = float(accumulator.hit.float().mean().item()) if len(points) else 0.0
    field = OverlapField(len(points), hypotheses, instance_features)
    metrics = {
        "grid": int(values["grid"]),
        "descriptor_dimension": backbone.dimension,
        "selected_frames": len(selected_frames),
        "valid_descriptor_count": int(
            np.linalg.norm(instance_features, axis=1).astype(bool).sum()
        ),
        "direct_point_hit_fraction": hit_fraction,
        "instance_field_coverage": float(field.covered.mean()) if len(points) else 0.0,
    }
    return instance_features, metrics


@torch.no_grad()
def pool_instance_descriptors(
    sum_wf: torch.Tensor,
    sum_w: torch.Tensor,
    hypotheses: Sequence[np.ndarray],
    *,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Return one float16 unit descriptor per hypothesis, or a zero row when unobserved."""
    if sum_wf.ndim != 2 or sum_w.shape != (sum_wf.shape[0],):
        raise ValueError("Expected sum_wf (N, D) and sum_w (N,)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    descriptors = torch.zeros(
        len(hypotheses), sum_wf.shape[1], dtype=torch.float32, device=sum_wf.device
    )
    point_count = sum_wf.shape[0]
    for hypothesis_id, hypothesis in enumerate(hypotheses):
        hypothesis = np.asarray(hypothesis, dtype=np.int64)
        if hypothesis.ndim != 1 or ((hypothesis < 0) | (hypothesis >= point_count)).any():
            raise ValueError(f"Hypothesis {hypothesis_id} contains invalid point indices")
        feature_sum = torch.zeros(sum_wf.shape[1], dtype=torch.float32, device=sum_wf.device)
        observed_count = 0
        for start in range(0, len(hypothesis), chunk_size):
            indices = torch.as_tensor(
                hypothesis[start : start + chunk_size], dtype=torch.long, device=sum_wf.device
            )
            indices = indices[sum_w[indices] > 0]
            if not indices.numel():
                continue
            point_features = sum_wf[indices] / sum_w[indices, None]
            feature_sum += _l2_torch(point_features).sum(0)
            observed_count += indices.numel()
        if observed_count:
            descriptors[hypothesis_id] = _l2_torch(feature_sum / observed_count)
    return descriptors.to(torch.float16).cpu().numpy()


class OverlapField:
    """Bounded reconstruction of the normalized mean descriptor at each covered point."""

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
        result[observed] = _l2_numpy(result[observed])
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


def write_semantic_pca_ply(
    path: str | Path,
    points: np.ndarray,
    normals: np.ndarray,
    hypotheses: Sequence[np.ndarray],
    descriptors: np.ndarray,
    *,
    chunk_size: int = 65536,
    max_pca_samples: int = 100000,
) -> float:
    """Write a deterministic PCA-colored overlap field and return its point coverage."""
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("Points and normals must have matching (N, 3) shapes")
    if max_pca_samples <= 0:
        raise ValueError("max_pca_samples must be positive")

    field = OverlapField(len(points), hypotheses, descriptors)
    covered_indices = np.flatnonzero(field.covered)
    covered_count = len(covered_indices)
    mean = np.zeros(field.descriptors.shape[1], np.float64)
    for _, values, covered in field.chunks(chunk_size):
        mean += values[covered].sum(0, dtype=np.float64)
    if covered_count:
        mean = (mean / covered_count).astype(np.float32)
        sample_indices = np.random.default_rng(0).choice(
            covered_indices, min(covered_count, max_pca_samples), replace=False
        )
        sample = field.rows(sample_indices)
        _, _, right = np.linalg.svd(sample - mean, full_matrices=False)
        components = np.zeros((3, field.descriptors.shape[1]), np.float32)
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
        for point_slice, values, covered in field.chunks(chunk_size):
            records = np.empty(point_slice.stop - point_slice.start, record_dtype)
            records["x"], records["y"], records["z"] = points[point_slice].T
            records["nx"], records["ny"], records["nz"] = normals[point_slice].T
            colors = np.full((len(records), 3), 128, np.uint8)
            if covered.any():
                projection = (values[covered] - mean) @ components.T
                colors[covered] = (
                    np.clip((projection - low) / (high - low + 1e-8), 0, 1) * 255
                ).astype(np.uint8)
            records["red"], records["green"], records["blue"] = colors.T
            handle.write(records.tobytes())
    return covered_count / len(points) if len(points) else 0.0
