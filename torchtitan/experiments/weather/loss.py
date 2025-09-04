from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def mae_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean absolute error"""
    return torch.mean(torch.abs(pred.float() - labels.float().detach()))

def build_mae_loss(job_config: JobConfig) -> LossFunction:
    loss_fn = mae_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn
