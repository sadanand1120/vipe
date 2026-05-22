import argparse

from pathlib import Path

import hydra

from vipe import get_config_path
from vipe.pipeline import VipePipeline
from vipe.streams.base import FrameDir
from vipe.utils.determinism import seed_everything
from vipe.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ViPE on a frame directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for saved artifacts")
    return parser


def compose_config(overrides: list[str]):
    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        return hydra.compose("default", overrides=overrides)


def main() -> None:
    cli_args, hydra_overrides = build_parser().parse_known_args()
    cfg = compose_config(hydra_overrides)
    seed_everything(cfg.seed)
    logger = configure_logging()
    pipeline = VipePipeline(
        slam=cfg.pipeline.slam,
        output=cfg.pipeline.output,
        output_dir=cli_args.output_dir,
    )
    frame_stream = FrameDir(
        path=cfg.streams.base_path,
        fps=cfg.streams.fps,
    )
    logger.info(f"Processing {frame_stream.name()}")
    pipeline.run(frame_stream)
    logger.info(f"Finished processing {frame_stream.name()}")


if __name__ == "__main__":
    main()
