import hydra
from omegaconf import DictConfig

from vipe.runtime import make_annotation_pipeline, make_frame_dir_stream_list
from vipe.utils.logging import configure_logging


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    stream_list = make_frame_dir_stream_list(args.streams)

    logger = configure_logging()
    pipeline = make_annotation_pipeline(args.pipeline)
    for stream_idx in range(len(stream_list)):
        video_stream = stream_list[stream_idx]
        logger.info(
            f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
        )
        pipeline.run(video_stream)
        logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
