from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from typing import Any
from .dataset.aardvark_dataset import build_weather_dataloader
from .models.simple_llama3.model.model import SimpleLlama3Model
from .models.simple_llama3.infra.parallelize import parallelize_llama
from .models.simple_llama3.infra.pipeline import pipeline_llama

# needed
name = "llama3-weather"
model_cls = SimpleLlama3Model
build_loss_function = None
build_tokenizer_fn = None
build_validator_fn = None
state_dict_adapter = None
build_tokenizer_fn = None
build_validator_fn = None
state_dict_adapter = None
model_configs: dict[str, Any] = {}

register_train_spec(
    TrainSpec(
        name="llama3-weather",
        model_cls=SimpleLlama3Model,  # TODO
        model_args=model_configs,  # TODO
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_weather_dataloader,
        build_tokenizer_fn=None,  # TODO
        build_loss_fn=build_loss_function,  # TODO
        build_validator_fn=build_flux_validator,  # TODO
        state_dict_adapter=FluxStateDictAdapter,  # TODO
    )
)

from .dataset.flux_dataset import build_flux_dataloader
from .infra.parallelize import parallelize_flux
from .loss import build_mse_loss
from .model.args import FluxModelArgs
from .model.autoencoder import AutoEncoderParams
from .model.model import FluxModel
from .model.state_dict_adapter import FluxStateDictAdapter
from .validate import build_flux_validator

__all__ = [
    "FluxModelArgs",
    "FluxModel",
    "flux_configs",
    "parallelize_flux",
]


flux_configs = {
    "flux-dev": FluxModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": FluxModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-debug": FluxModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=1536,
        mlp_ratio=4.0,
        num_heads=12,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}
