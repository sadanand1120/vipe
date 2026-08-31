#!/usr/bin/env python3

from pathlib import Path

from vipe import get_config_path
from vipe.bench.instance_benchmark import InstanceBenchmarkSpec, run_instance_benchmark
from vipe.bench.scannet import ScanNetDataset
from vipe.bench.scannet_instance import evaluate_scene


if __name__ == "__main__":
    config_root = get_config_path()
    run_instance_benchmark(
        InstanceBenchmarkSpec(
            dataset_key="scannet",
            dataset_label="ScanNet",
            script_path=Path(__file__),
            pipeline_config_path=config_root / "default.yaml",
            instance_config_path=config_root / "default_instance.yaml",
            eval_config_path=config_root / "eval_scannet_instance_config.yaml",
            dataset_type=ScanNetDataset,
            evaluate_scene=evaluate_scene,
        )
    )
