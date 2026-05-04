import hydra
from omegaconf import DictConfig

from vipe.pipeline.default import VipePipeline
from vipe.streams.base import FrameDir
from vipe.utils.logging import configure_logging


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    logger = configure_logging()
    frame_stream = FrameDir(
        path=args.streams.base_path,
        fps=args.streams.fps,
        frame_start=args.streams.frame_start,
        frame_end=args.streams.frame_end,
        frame_skip=args.streams.frame_skip,
    )
    pipeline = VipePipeline(
        slam=args.pipeline.slam,
        output=args.pipeline.output,
    )
    logger.info(f"Processing {frame_stream.name()}")
    pipeline.run(frame_stream)
    logger.info(f"Finished processing {frame_stream.name()}")


if __name__ == "__main__":
    run()
