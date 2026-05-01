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

import click
import hydra

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing image frames",
)
@click.option(
    "--image-dir-fps",
    type=float,
    help="Frame rate for --image-dir input",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
def infer(video: Path, image_dir: Path, image_dir_fps: float | None, output: Path, pipeline: str, visualize: bool):
    """Run inference on a video file or directory of images."""

    logger = configure_logging()

    # Validate that exactly one input source is provided
    if not video and not image_dir:
        click.echo("Error: Must provide either a video file or --image-dir", err=True)
        raise click.Abort()
    
    if video and image_dir:
        click.echo("Error: Cannot provide both video file and --image-dir", err=True)
        raise click.Abort()

    if image_dir and image_dir_fps is None:
        click.echo("Error: Must provide --image-dir-fps when using --image-dir", err=True)
        raise click.Abort()

    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={output}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    # Set up stream configuration based on input type
    if image_dir:
        overrides.extend([
            "streams=frame_dir_stream",
            f"streams.base_path={image_dir}"
        ])
        input_path = image_dir
        input_desc = f"image directory {image_dir}"
    else:
        input_path = video
        input_desc = f"video {video}"

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {input_desc}...")
    vipe_pipeline = make_pipeline(args.pipeline)

    if image_dir:
        # Use frame directory stream
        video_stream = ProcessedVideoStream(
            FrameDirStream(image_dir, fps=image_dir_fps),  # type: ignore[arg-type]
            [],
        )
    else:
        # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
        video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
@click.option("--host", default="0.0.0.0", type=str, help="Host interface to bind the visualization server to")
@click.option("--share/--no-share", default=False, help="Request a public share URL from viser")
def visualize(data_path: Path, port: int, host: str, share: bool):
    run_viser(data_path, port, host, share)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
