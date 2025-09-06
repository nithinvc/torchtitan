# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import torch
import torch.nn as nn

from torchtitan.components.loss import LossFunction
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.train_spec import BaseModelArgs, ParallelizeFunction

"""
Adapted from the llama3 example model.
Simple llama3 doesn't support pp
"""


def pipeline_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_args: BaseModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[object | None, list[nn.Module], bool, bool]:
    # This should never be called for now

    assert not parallel_dims.pp_enabled, "Simple llama3 doesn't support pp"

    # No pipeline parallelism: apply SPMD-style parallelization to the whole model
    model = parallelize_fn(model, parallel_dims, job_config)

    # In the non-PP path, treat the single model as both first and last stage
    return None, [model], True, True
