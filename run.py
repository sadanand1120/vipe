import hydra
from omegaconf import DictConfig

from vipe.pipeline import VipePipeline
from vipe.streams.base import FrameDir
from vipe.utils.determinism import seed_everything
from vipe.utils.logging import configure_logging


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    seed_everything(args.seed)
    logger = configure_logging()
    pipeline = VipePipeline(
        slam=args.pipeline.slam,
        output=args.pipeline.output,
    )
    frame_stream = FrameDir(
        path=args.streams.base_path,
        fps=args.streams.fps,
    )
    logger.info(f"Processing {frame_stream.name()}")
    pipeline.run(frame_stream)
    logger.info(f"Finished processing {frame_stream.name()}")


if __name__ == "__main__":
    run()
