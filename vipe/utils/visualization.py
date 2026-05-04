# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
from pycg import image

from vipe.ext.lietorch import SE3
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import FrameData, FrameStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional

from .geometry import project_points_to_panorama, project_points_to_pinhole


POINTS_STENCIL = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
POINTS_STENCIL = np.stack(POINTS_STENCIL, axis=-1).reshape(-1, 2)
POINTS_STENCIL = POINTS_STENCIL[np.max(np.abs(POINTS_STENCIL), axis=-1) > 1]
POINTS_STENCIL = np.pad(POINTS_STENCIL, ((0, 1), (0, 0)), constant_values=0)


class VideoWriter:
    """
    Simple video writer class (use h264 codec).

    Usage:
    ```
    with VideoWriter("output.mp4", 30) as vw:
        for frame in frames:
            vw.write(frame)
    ```
    """

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.vw: Any = None

    def __enter__(self):
        return self

    def write(self, frame: np.ndarray):
        if self.vw is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.vw = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", macro_block_size=None)

        if frame.dtype in [np.float32, np.float64]:
            frame = (frame * 255).astype(np.uint8)

        assert self.vw is not None
        self.vw.append_data(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.vw is not None:
            self.vw.close()


def bbox_with_size(pcd_xyz: torch.Tensor, quantile: float = 0.98):
    from pycg import vis

    low_quantile, high_quantile = (1 - quantile) / 2, 1 - (1 - quantile) / 2
    pcd_min = torch.quantile(pcd_xyz, low_quantile, dim=0, keepdim=True)
    pcd_max = torch.quantile(pcd_xyz, high_quantile, dim=0, keepdim=True)

    x_length = pcd_max[0, 0] - pcd_min[0, 0]
    x_length_pos = pcd_min[0] + torch.tensor([x_length / 2, 0, 0])
    y_length = pcd_max[0, 1] - pcd_min[0, 1]
    y_length_pos = pcd_min[0] + torch.tensor([0, y_length / 2, 0])
    z_length = pcd_max[0, 2] - pcd_min[0, 2]
    z_length_pos = pcd_min[0] + torch.tensor([0, 0, z_length / 2])

    return [
        vis.wireframe_bbox(pcd_min, pcd_max, ucid=-1),
        vis.text(f"{x_length.item():.2f}m", x_length_pos),
        vis.text(f"{y_length.item():.2f}m", y_length_pos),
        vis.text(f"{z_length.item():.2f}m", z_length_pos),
    ]


def colorize_depth(
    depth: np.ndarray,
    normalize: bool = False,
    clip_depth: bool = False,
    min_depth: float = 1e-3,
    max_depth: float = 1e4,
):
    if clip_depth:
        depth = np.clip(depth, a_min=min_depth, a_max=max_depth)

    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min())

    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth


def draw_points_batch(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: np.ndarray | tuple | None = None,
    stencil: np.ndarray | None = None,
):
    if stencil is None:
        stencil = np.array([[0, 0]])

    for rel_pos in stencil:
        pos = (pts + rel_pos[None]).astype(int)
        in_bound = (pos[:, 0] >= 0) & (pos[:, 0] < canvas.shape[1]) & (pos[:, 1] >= 0) & (pos[:, 1] < canvas.shape[0])
        pos = pos[in_bound]
        if isinstance(color, np.ndarray):
            canvas[pos[:, 1], pos[:, 0]] = color[in_bound]
        else:
            canvas[pos[:, 1], pos[:, 0]] = color or (0, 255, 0)
    return canvas


def draw_lines_batch(
    canvas: np.ndarray,
    lines_start: np.ndarray,
    lines_end: np.ndarray,
    color: tuple | None = None,
):
    if lines_start.shape[0] == 0:
        return canvas
    lines = np.stack([lines_start, lines_end], axis=1).astype(int)
    return cv2.polylines(
        canvas.copy(),
        [l for l in lines],
        isClosed=False,
        color=color or (0, 255, 0),
        thickness=1,
    )


def draw_tracks(canvas: np.ndarray, tracks: np.ndarray, valid: np.ndarray):
    """
    Args:
        canvas: The image to draw the tracks on. (H, W, 3) uint8
        tracks: The tracks to draw. (length, n_tracks, 2)
            To draw tracks of different lengths, please call this function multiple times.
        valid: The validity of the tracks. (length, n_tracks)
    """
    for l in range(tracks.shape[0]):
        uv, uv_valid = tracks[l], valid[l]
        canvas = draw_points_batch(canvas, uv[uv_valid], (0, 255 - 20 * l, 0), stencil=POINTS_STENCIL)
    for l in range(tracks.shape[0] - 1):
        uv_start, start_valid = tracks[l], valid[l]
        uv_end, end_valid = tracks[l + 1], valid[l + 1]
        all_valid = start_valid & end_valid
        canvas = draw_lines_batch(canvas, uv_start[all_valid], uv_end[all_valid], (0, 255 - 20 * l, 0))
    return canvas


def project_points_panorama(
    xyz: np.ndarray,
    pose: SE3,
    frame_size: tuple[int, int],
    color: np.ndarray | None = None,
) -> np.ndarray:
    assert pose.shape == (), "Only single pose is supported"

    canvas = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255

    pose_matrix = pose.inv().matrix().cpu().numpy()
    local_xyz = xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    uv = project_points_to_panorama(torch.from_numpy(local_xyz), return_depth=False)
    uv[:, 0] *= frame_size[1]
    uv[:, 1] *= frame_size[0]
    uv = (uv - 0.5).round().int().cpu().numpy()

    if color is not None:
        if np.issubdtype(color.dtype, np.floating):
            color = (color * 255).astype(np.uint8)

    return draw_points_batch(canvas, uv, color, stencil=POINTS_STENCIL)


def project_points(
    xyz: np.ndarray,
    intrinsics: np.ndarray,
    camera_type: CameraType,
    pose: SE3,
    frame_size: tuple[int, int],
    subsample_factor: int,
    color: np.ndarray | None = None,
) -> np.ndarray:
    assert pose.shape == (), "Only single pose is supported"

    canvas = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255

    pose_matrix = pose.inv().matrix().cpu().numpy()
    local_xyz = xyz @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]

    camera_model = camera_type.build_camera_model(torch.from_numpy(intrinsics)).scaled(1.0 / subsample_factor)
    xyz_h = torch.cat([torch.from_numpy(local_xyz), torch.ones((local_xyz.shape[0], 1))], dim=1)
    uv, _, _ = camera_model.proj_points(xyz_h)
    in_bound = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < frame_size[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < frame_size[0])
        & torch.from_numpy(local_xyz[:, 2] > 0)
    )
    uv = uv[in_bound]
    uv = (uv - 0.5).round().int().cpu().numpy()

    # uv, in_bound = project_points_to_pinhole(
    #     torch.from_numpy(local_xyz),
    #     torch.from_numpy(intrinsics),
    #     frame_size,
    #     return_depth=False,
    # )
    # uv = uv[in_bound]
    # uv[:, 0] *= frame_size[1]
    # uv[:, 1] *= frame_size[0]
    # uv = (uv - 0.5).round().int().cpu().numpy()

    if color is not None:
        color = color[in_bound]
        if np.issubdtype(color.dtype, np.floating):
            color = (color * 255).astype(np.uint8)

    return draw_points_batch(canvas, uv, color, stencil=POINTS_STENCIL)


def image_above_text(img: np.ndarray, text: str = "<TEXT>") -> Image.Image:
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)

    width, height = image.size
    text_height = max(20, height // 10)

    new_height = height + int(text_height * 1.5)
    new_image = Image.new("RGB", (width, new_height), color=(255, 255, 255))
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)

    try:
        font = ImageFont.truetype("arial.ttf", text_height)  # You can change the font size
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial is not available

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    text_y = height + int(text_height * 0.2)

    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  # Black text
    return new_image


def save_projection_video(
    video_path: Path,
    frame_stream: FrameStream,
    slam_output: SLAMOutput | None,
    subsample_factor: int,
    attributes: list[list[str]],
):
    img_h, img_w = frame_stream.frame_size()
    img_h //= subsample_factor
    img_w //= subsample_factor

    na_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    text_img = image.text("N/A")
    na_img = image.place_image(
        text_img,
        na_img,
        img_w // 2 - text_img.shape[1] // 2,
        img_h // 2 - text_img.shape[0] // 2,
    )
    na_img = (na_img[..., :3] * 255).astype(np.uint8)

    slam_map = unpack_optional(slam_output.slam_map) if slam_output is not None and slam_output.slam_map is not None else None
    if slam_map is not None:
        pcd_xyz = slam_map.dense_disp_xyz.cpu().numpy()
        pcd_rgb = slam_map.dense_disp_rgb.cpu().numpy()
    else:
        pcd_xyz = None
        pcd_rgb = None

    depth_range = [np.inf, -np.inf]
    rectified_coords_norm = None

    def get_rgb_img(frame_data: FrameData) -> np.ndarray:
        rgb_img = frame_data.rgb.cpu().numpy().astype(float)
        rgb_img = (rgb_img * 255).astype(np.uint8)
        return cv2.resize(rgb_img, (img_w, img_h))

    def get_depth_img(frame_data: FrameData) -> np.ndarray:
        if frame_data.metric_depth is None:
            return na_img

        depth_data = frame_data.metric_depth.reciprocal()
        valid_depth = depth_data[torch.isfinite(depth_data)]
        if valid_depth.numel() == 0:
            return na_img

        depth_min_q, depth_max_q = torch.quantile(valid_depth, torch.tensor([0.05, 0.95], device=valid_depth.device))
        depth_range[0] = min(depth_range[0], depth_min_q.item())
        depth_range[1] = max(depth_range[1], depth_max_q.item())
        depth_middle = (depth_range[0] + depth_range[1]) / 2
        depth_scale = max(depth_range[1] - depth_range[0], 1e-6)
        depth_min = depth_middle - depth_scale / 2 * 1.3
        depth_max = depth_middle + depth_scale / 2 * 1.3

        depth_data = depth_data.clone()
        depth_data[~torch.isfinite(depth_data)] = depth_min
        depth_data = depth_data[::subsample_factor, ::subsample_factor]
        depth_img = depth_data.cpu().numpy().astype(float)
        depth_img = (depth_img - depth_min) / (depth_max - depth_min)
        return colorize_depth(np.clip(depth_img, 0, 1))

    def get_pcd_img(frame_data: FrameData, rgb_img: np.ndarray) -> np.ndarray:
        if slam_map is None:
            return na_img

        intrinsics = unpack_optional(frame_data.intrinsics)
        if torch.sum(intrinsics) < 1e-6:
            pcd_img = project_points_panorama(
                unpack_optional(pcd_xyz),
                frame_data.pose,
                frame_size=(img_h, img_w),
                color=pcd_rgb,
            )
        else:
            pcd_img = project_points(
                unpack_optional(pcd_xyz),
                frame_data.intrinsics.cpu().numpy(),
                camera_type=frame_data.camera_type,
                pose=frame_data.pose,
                frame_size=(img_h, img_w),
                subsample_factor=subsample_factor,
                color=pcd_rgb,
            )
        return cv2.addWeighted(rgb_img, 0.2, pcd_img, 0.8, 0)

    def get_rectified_img(frame_data: FrameData) -> np.ndarray:
        nonlocal rectified_coords_norm
        if frame_data.intrinsics is None or frame_data.camera_type is None:
            return na_img

        if rectified_coords_norm is None:
            original_intr = frame_data.camera_type.build_camera_model(frame_data.intrinsics).scaled(1 / subsample_factor)
            pinhole_intr = original_intr.pinhole()
            device = pinhole_intr.intrinsics.device
            y, x = torch.meshgrid(torch.arange(img_h).float(), torch.arange(img_w).float(), indexing="ij")
            y, x = y.to(device), x.to(device)
            pts, _, _ = pinhole_intr.iproj_disp(torch.ones_like(x), x, y)
            coords, _, _ = original_intr.proj_points(pts)
            rectified_coords_norm = (2.0 * coords / torch.tensor([img_w, img_h], device=coords.device) - 1.0).reshape(
                1, img_h, img_w, 2
            )

        img = frame_data.rgb.permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.grid_sample(
            img,
            rectified_coords_norm,
            mode="bilinear",
            align_corners=False,
        )[0].float()
        return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    with VideoWriter(video_path, frame_stream.fps()) as vw:
        trajectory_length = 0.0
        last_pose = None
        for frame_idx, frame_data in pbar(enumerate(frame_stream), total=len(frame_stream), desc="Writing viz video"):
            rgb_img = get_rgb_img(frame_data)
            img_rows = []
            for attr_row in attributes:
                img_row = []
                for attr_name in attr_row:
                    if attr_name == "rgb":
                        img_row.append(rgb_img)
                    elif attr_name == "depth":
                        img_row.append(get_depth_img(frame_data))
                    elif attr_name == "pcd":
                        img_row.append(get_pcd_img(frame_data, rgb_img))
                    elif attr_name == "rectified":
                        img_row.append(get_rectified_img(frame_data))
                    elif attr_name == "empty":
                        img_row.append(na_img)
                    else:
                        raise ValueError(f"Unknown visualization attribute: {attr_name}")
            img_rows.append(np.concatenate(img_row, axis=1))
            img_final = np.concatenate(img_rows, axis=0)
            text_desc = f"Frame {frame_idx:03d}"
            if frame_data.intrinsics is not None:
                focal = frame_data.intrinsics[0].item()
                if focal > 1e-6:  # Pano has focal 0
                    fov_y = 2 * np.arctan(frame_data.size()[0] / (2 * focal))
                    fov_y = np.rad2deg(fov_y)
                    text_desc += f" | fovY {fov_y:.2f}"
            current_pose = frame_data.pose
            if current_pose is not None:
                if last_pose is not None:
                    trajectory_length += np.linalg.norm((last_pose.inv() * current_pose).translation()[:3].cpu().numpy())
                last_pose = current_pose
            text_desc += f" | Traj {trajectory_length:.4f}"
            if len(frame_data.information) > 0:
                text_desc += f" | {frame_data.information}"
            img_text = image.text(text_desc)
            img_final = image.place_image(img_text, img_final, 0, 0)
            vw.write(img_final)
