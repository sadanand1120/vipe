import hydra
from omegaconf import DictConfig

from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.pipeline.processors import ScanNetGTProcessor
from vipe.streams.base import FrameDir, ProcessedFrameStream
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
    output_stream = frame_stream
    if args.pipeline.use_gt_pose or args.pipeline.use_gt_depth:
        output_stream = ProcessedFrameStream(
            frame_stream,
            [
                ScanNetGTProcessor(
                    frame_stream.frame_files,
                    frame_stream.path.parent,
                    use_gt_pose=args.pipeline.use_gt_pose,
                    use_gt_depth=args.pipeline.use_gt_depth,
                )
            ],
        )

    pipeline = DefaultAnnotationPipeline(
        init=args.pipeline.init,
        slam=args.pipeline.slam,
        post=args.pipeline.post,
        output=args.pipeline.output,
        use_gt_pose=args.pipeline.use_gt_pose,
        use_gt_depth=args.pipeline.use_gt_depth,
    )
    logger.info(f"Processing {frame_stream.name()}")
    pipeline.run(output_stream, source_frame_dir=frame_stream.path)
    logger.info(f"Finished processing {frame_stream.name()}")


if __name__ == "__main__":
    run()
