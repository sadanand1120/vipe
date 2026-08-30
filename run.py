import argparse

from pathlib import Path

from vipe import get_config_path
from vipe.pipeline import VipePipeline
from vipe.stream import FrameDir
from vipe.utils.config import load_yaml_config
from vipe.utils.determinism import seed_everything
from vipe.utils.logging import configure_logging


DEFAULT_CONFIG_PATH = get_config_path() / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ViPE on a canonical RGB-D scene directory.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Canonical ViPE RGB-D scene directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for saved artifacts")
    return parser


def main() -> None:
    cli_args = build_parser().parse_args()
    cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
    seed_everything(cfg.seed, temporary_determinism=cfg.temporary_determinism)
    logger = configure_logging()
    pipeline = VipePipeline(
        slam=cfg.pipeline.slam,
        output=cfg.pipeline.output,
        output_dir=cli_args.output_dir,
    )
    frame_stream = FrameDir(cli_args.input_dir)
    logger.info("Processing %s (%d canonical frames)", frame_stream.name, len(frame_stream))
    pipeline.run(frame_stream)
    logger.info(f"Finished processing {frame_stream.name}")


if __name__ == "__main__":
    main()
