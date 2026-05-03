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

from omegaconf import DictConfig

from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.streams.frame_dir_stream import FrameDirStreamList


def make_frame_dir_stream_list(config: DictConfig) -> FrameDirStreamList:
    return FrameDirStreamList(
        base_path=config.base_path,
        fps=config.fps,
        frame_start=config.frame_start,
        frame_end=config.frame_end,
        frame_skip=config.frame_skip,
        cached=config.cached,
    )


def make_annotation_pipeline(config: DictConfig) -> DefaultAnnotationPipeline:
    return DefaultAnnotationPipeline(
        init=config.init,
        slam=config.slam,
        post=config.post,
        output=config.output,
    )
