"""Streaming SAM1 seeds, SAM2 propagation, and deterministic 3D track linking."""

import shutil
import tempfile
import time

from collections.abc import Callable, Sequence
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from vipe.instance.lift import lift_masks


class _UnionFind:
    """Union-find whose representative is always the minimum track ID."""

    def __init__(self) -> None:
        self.parents: dict[int, int] = {}

    def find(self, value: int) -> int:
        while self.parents.setdefault(value, value) != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def _encode(mask: np.ndarray) -> bytes:
    from pycocotools import mask as mask_utils

    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]


def _decode(counts: bytes, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    return mask_utils.decode({"size": [height, width], "counts": counts}).astype(bool)


class SAM1Generator:
    """Frozen SAM1 ViT-H automatic mask generator."""

    def __init__(self, config, device: torch.device) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        values = dict(config)
        model_path = str(values.pop("model_path"))
        model = sam_model_registry["vit_h"](checkpoint=model_path).to(device).eval()
        self.generator = SamAutomaticMaskGenerator(
            model,
            output_mode="binary_mask",
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=0.6,
            min_mask_region_area=0,
            **values,
        )

    def generate(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        with torch.no_grad():
            masks = self.generator.generate(rgb)
        return [
            (np.asarray(mask["segmentation"], dtype=bool), float(mask["predicted_iou"]))
            for mask in masks
        ]


class SAM2Tracker:
    """One SAM2.1-small predictor reused across every scene chunk."""

    def __init__(self, config, device: torch.device) -> None:
        from sam2.build_sam import build_sam2_video_predictor
        from tqdm import tqdm

        values = dict(config)
        self.threshold = float(values["threshold"])
        self.device = device
        self.predictor = build_sam2_video_predictor(
            config_file=str(values["model_config"]),
            ckpt_path=str(values["model_path"]),
            device=str(device),
            apply_postprocessing=False,
        )
        # SAM2 has no progress toggle; ViPE owns one scene-level lift bar instead.
        silent_tqdm = partial(tqdm, disable=True)
        import_module(type(self.predictor).__module__).tqdm = silent_tqdm
        import_module("sam2.utils.misc").tqdm = silent_tqdm

    def track(
        self, frame_directory: Path, frame_count: int, seeds: Sequence[tuple[int, np.ndarray]]
    ) -> list[dict[int, np.ndarray]]:
        """Propagate all seed objects in one state; frame zero keeps verbatim SAM1 masks."""
        output: list[dict[int, np.ndarray]] = [dict() for _ in range(frame_count)]
        live = []
        for object_id, mask in seeds:
            if mask.any():
                output[0][object_id] = mask
                live.append((object_id, mask))
        if not live or frame_count < 2:
            return output
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(
                video_path=str(frame_directory),
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )
            for object_id, mask in live:
                self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(object_id),
                    mask=torch.from_numpy(mask.astype(np.uint8)),
                )
            for frame_index, object_ids, logits in self.predictor.propagate_in_video(state):
                if frame_index == 0:
                    continue
                predicted = (logits[:, 0] > self.threshold).cpu().numpy()
                for index, object_id in enumerate(object_ids):
                    if predicted[index].any():
                        output[frame_index][int(object_id)] = predicted[index]
            self.predictor.reset_state(state)
            del state
        torch.cuda.empty_cache()
        return output


class StreamingMasks:
    """Generate one chunk at a time and expose masks in ascending retained-frame order."""

    def __init__(
        self,
        keyframes: Sequence[int],
        rgb_of: Callable[[int], np.ndarray],
        width: int,
        height: int,
        config,
        jpeg_quality: int,
        device: torch.device,
    ) -> None:
        self.keyframes = [int(frame) for frame in keyframes]
        self.position = {frame: index for index, frame in enumerate(self.keyframes)}
        self.rgb_of = rgb_of
        self.width, self.height = width, height
        self.chunk_size = int(config["chunk_keyframes"])
        self.seed_topk = int(config["seed_topk"])
        self.stitch_config = config["stitch"]
        self.sam1_config = config["sam1"]
        self.sam2_config = config["sam2"]
        self.jpeg_quality = int(jpeg_quality)
        self.device = device
        self.frame_masks: dict[int, list[dict]] = {}
        self.built_chunk = -1
        self.next_track_id = 0
        self.chunk_starts: list[int] = []
        self.union_find = _UnionFind()
        self.sam1 = None
        self.sam2 = None
        self.stats = {
            "seed_frames": 0,
            "seed_masks": 0,
            "tracked_masks": 0,
            "linked_tracks": 0,
            "low_iou": 0,
            "ambiguous": 0,
        }
        self.timings = {"model_build_s": 0.0, "sam1_s": 0.0, "sam2_s": 0.0}

    def _models(self):
        if self.sam1 is None:
            start = time.perf_counter()
            self.sam1 = SAM1Generator(self.sam1_config, self.device)
            self.sam2 = SAM2Tracker(self.sam2_config, self.device)
            self.timings["model_build_s"] += time.perf_counter() - start
        return self.sam1, self.sam2

    def _build_chunk(self, chunk_index: int) -> None:
        from PIL import Image

        sam1, sam2 = self._models()
        start = chunk_index * self.chunk_size
        chunk = self.keyframes[start : start + self.chunk_size]
        directory = Path(tempfile.mkdtemp(prefix="vipe-instance-chunk-"))
        try:
            for index, frame in enumerate(chunk):
                Image.fromarray(self.rgb_of(frame)).save(
                    directory / f"{index}.jpg", quality=self.jpeg_quality
                )
            seed_rgb = np.asarray(Image.open(directory / "0.jpg").convert("RGB"))
            tick = time.perf_counter()
            seeds = sam1.generate(seed_rgb)
            self.timings["sam1_s"] += time.perf_counter() - tick
            if len(seeds) > self.seed_topk:
                seeds = sorted(seeds, key=lambda item: -item[1])[: self.seed_topk]
            base = self.next_track_id
            self.chunk_starts.append(base)
            self.next_track_id += len(seeds)
            tracked_seeds = [(base + offset, mask) for offset, (mask, _) in enumerate(seeds)]
            scores = {base + offset: score for offset, (_, score) in enumerate(seeds)}

            tick = time.perf_counter()
            propagated = sam2.track(directory, len(chunk), tracked_seeds)
            self.timings["sam2_s"] += time.perf_counter() - tick
        finally:
            shutil.rmtree(directory)
        for frame_index, frame_masks in zip(chunk, propagated):
            entries = [
                {"counts": _encode(mask), "score": scores[track_id], "track_id": int(track_id)}
                for track_id, mask in sorted(frame_masks.items())
            ]
            self.frame_masks[frame_index] = entries
            self.stats["tracked_masks"] += len(entries)
        self.stats["seed_frames"] += 1
        self.stats["seed_masks"] += len(seeds)

    def _ensure_chunk(self, chunk_index: int) -> None:
        while self.built_chunk < chunk_index:
            next_chunk = self.built_chunk + 1
            self._build_chunk(next_chunk)
            if self.built_chunk >= 0:
                start = self.built_chunk * self.chunk_size
                for frame in self.keyframes[start : start + self.chunk_size]:
                    self.frame_masks.pop(frame, None)
            self.built_chunk = next_chunk

    def masks_of(self, frame_index: int):
        frame_index = int(frame_index)
        self._ensure_chunk(self.position[frame_index] // self.chunk_size)
        return [
            (
                _decode(entry["counts"], self.height, self.width),
                entry["score"],
                entry["track_id"],
                entry["track_id"],
            )
            for entry in self.frame_masks.get(frame_index, [])
        ]

    def _link_tracks(self, track_unions: dict[int, np.ndarray]) -> None:
        import scipy.sparse as sp

        if not track_unions:
            return
        track_ids = np.array(sorted(track_unions), np.int64)
        voxel_sets = [track_unions[int(track_id)] for track_id in track_ids]
        sizes = np.array([len(voxels) for voxels in voxel_sets], np.int64)
        chunk_starts = np.asarray(self.chunk_starts, np.int64)
        chunks = np.searchsorted(chunk_starts, track_ids, side="right") - 1

        columns = np.concatenate(voxel_sets)
        rows = np.repeat(np.arange(len(track_ids), dtype=np.int64), sizes)
        incidence = sp.csr_matrix(
            (np.ones(len(columns), np.float32), (rows, columns)),
            shape=(len(track_ids), int(columns.max()) + 1),
        )
        min_iou = float(self.stitch_config["min_iou"])
        margin = float(self.stitch_config["margin"])
        best = {}
        for begin in range(0, len(track_ids), 512):
            intersections = (incidence[begin : begin + 512] @ incidence.T).tocsr()
            for row in range(intersections.shape[0]):
                source = begin + row
                lo, hi = intersections.indptr[row : row + 2]
                partners = intersections.indices[lo:hi]
                overlap = intersections.data[lo:hi]
                keep = chunks[partners] != chunks[source]
                partners, overlap = partners[keep], overlap[keep]
                if not partners.size:
                    continue
                iou = overlap / (sizes[source] + sizes[partners] - overlap)
                keep = iou >= min_iou
                if not keep.any():
                    self.stats["low_iou"] += 1
                    continue
                partners, iou = partners[keep], iou[keep]
                partner_chunks = chunks[partners]
                order = np.lexsort((-iou, partner_chunks))
                partners, iou, partner_chunks = partners[order], iou[order], partner_chunks[order]
                heads = np.flatnonzero(
                    np.r_[True, partner_chunks[1:] != partner_chunks[:-1]]
                )
                ends = np.r_[heads[1:], len(partner_chunks)]
                for head, end in zip(heads.tolist(), ends.tolist()):
                    runner_up = float(iou[head + 1]) if end - head > 1 else 0.0
                    best[(source, int(partner_chunks[head]))] = (
                        int(partners[head]),
                        float(iou[head]),
                        runner_up,
                    )

        pairs = []
        for (source, _), (partner, value, source_runner_up) in best.items():
            if partner < source:
                continue
            reverse = best.get((partner, int(chunks[source])))
            if reverse is None or reverse[0] != source:
                self.stats["ambiguous"] += 1
                continue
            if value < margin * source_runner_up or value < margin * reverse[2]:
                self.stats["ambiguous"] += 1
                continue
            pairs.append((value, source, partner))
        pairs.sort(reverse=True)

        component_chunks: dict[int, set[int]] = {}
        for _, source, partner in pairs:
            left = self.union_find.find(int(track_ids[source]))
            right = self.union_find.find(int(track_ids[partner]))
            if left == right:
                continue
            left_chunks = component_chunks.get(left) or {int(chunks[source])}
            right_chunks = component_chunks.get(right) or {int(chunks[partner])}
            if left_chunks & right_chunks:
                self.stats["ambiguous"] += 1
                continue
            self.union_find.union(left, right)
            component_chunks[self.union_find.find(left)] = left_chunks | right_chunks
            self.stats["linked_tracks"] += 1

    def finalize(self, evidence: dict, track_unions: dict[int, np.ndarray]) -> dict:
        self._link_tracks(track_unions)
        track_unions.clear()
        track_ids = evidence.pop("gm_track_id")
        evidence["global_track_ids"] = np.asarray(
            [self.union_find.find(int(track_id)) for track_id in track_ids],
            np.int64,
        )
        return evidence


def generate_and_lift(
    points: torch.Tensor,
    atom_of: np.ndarray,
    adjacency: tuple[np.ndarray, np.ndarray],
    keyframes: Sequence[int],
    rgb_of: Callable[[int], np.ndarray],
    depth_of: Callable[[int], np.ndarray],
    c2w_of: Callable[[int], np.ndarray],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    config,
    lift_config,
    jpeg_quality: int,
    device: torch.device,
):
    """Run the frozen mask, propagation, lift, and global-linking path."""
    masks = StreamingMasks(keyframes, rgb_of, width, height, config, jpeg_quality, device)
    evidence, track_unions = lift_masks(
        points,
        atom_of,
        adjacency,
        keyframes,
        c2w_of,
        masks.masks_of,
        depth_of,
        intrinsics,
        width,
        height,
        float(lift_config["occlusion_tolerance_m"]),
        int(lift_config["min_voxels"]),
    )
    masks.finalize(evidence, track_unions)
    global_tracks = int(np.unique(evidence["global_track_ids"]).size)
    return evidence, {**masks.stats, **masks.timings, "global_tracks": global_tracks}
