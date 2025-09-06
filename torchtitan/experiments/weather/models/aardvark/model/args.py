from dataclasses import dataclass

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs


@dataclass
class AardvarkModelArgs(BaseModelArgs):
    lead_time: int = 1
    return_gridded: bool = False  # TODO Not sure default or what they use

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        pass

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        return nparams, 2
