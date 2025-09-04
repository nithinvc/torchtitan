from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from typing import Any
from .dataset.aardvark_dataset import build_weather_dataloader
from .models.simple_llama3.model.model import SimpleLlama3Model
from .models.simple_llama3.infra.parallelize import parallelize_llama
from .models.simple_llama3.infra.pipeline import pipeline_llama
from .loss import build_mae_loss
from .models.simple_llama3.model.args import SimpleLlama3ModelArgs

from torchtitan.components.validate import build_validator

# needed
name = "llama3-weather"
model_cls = SimpleLlama3Model
build_tokenizer_fn = None
build_validator_fn = None
state_dict_adapter = None
build_tokenizer_fn = None
build_validator_fn = None
state_dict_adapter = None
model_configs: dict[str, Any] = {}

model_configs = {
    "llama3-debug": SimpleLlama3ModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2000, rope_theta=500000
    ),
}

register_train_spec(
    TrainSpec(
        name="llama3-weather",
        model_cls=SimpleLlama3Model,
        model_args=model_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_weather_dataloader,
        build_tokenizer_fn=None,  # TODO
        build_loss_fn=build_mae_loss,
        build_validator_fn=build_validator,  # TODO
        state_dict_adapter=None,
    )
)