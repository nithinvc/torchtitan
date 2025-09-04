from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from tensordict import TensorDictBase, tensorclass

from .aardvark_utils import broadcast_to_4d, to_channels_first_4d, to_channels_last_4d


@tensorclass
class ForecastTask:
    y_context: torch.Tensor
    lt: torch.Tensor


@tensorclass
class DownscalingTask:
    y_context: torch.Tensor
    x_context_lon: torch.Tensor
    x_context_lat: torch.Tensor
    x_target: torch.Tensor
    alt_target: torch.Tensor
    aux_time: torch.Tensor
    lt: torch.Tensor

    def as_model_input(self) -> Dict[str, Any]:
        return {
            "y_context": self.y_context,
            "x_context": [self.x_context_lon, self.x_context_lat],
            "x_target": self.x_target,
            "alt_target": self.alt_target,
            "aux_time": self.aux_time,
            "lt": self.lt,
        }


@tensorclass
class E2ETask:
    """
    Container for E2E inputs using tensorclass for non-mutating, typed access.

    - assimilation: a TensorDict with all tensors needed by the encoder (ConvCNPWeather in assimilation mode)
    - forecast: y_context and lt used by the processor
    - downscaling: fields used by the decoder
    """

    assimilation: TensorDictBase
    forecast: ForecastTask
    downscaling: DownscalingTask


class AardvarkE2E(nn.Module):
    """
    Major differences from original implementation:
    - Accepts pre-built submodules (encoder/processor/decoder) instead of loading from disk
    - Avoids in-place mutation of the input task; uses non-mutating transformations
    - Normalization factors are passed explicitly and registered as buffers for device movement
    - Grid sizes and channel slicing are configurable via constructor args
    """

    def __init__(
        self,
        *,
        se_model: nn.Module,
        forecast_models: Sequence[nn.Module],
        sf_model: nn.Module,
        lead_time: int,
        normalization: Dict[str, torch.Tensor],
        return_gridded: bool = False,
        # Hyperparameters extracted from original logic
        overwrite_channels: int = 24,
        base_context_exclude_tail: int = 11,
    ):
        super().__init__()

        if len(forecast_models) != lead_time:
            raise ValueError(f"Expected {lead_time} forecast models, got {len(forecast_models)}")

        self.se_model = se_model
        self.forecast_models = nn.ModuleList(list(forecast_models))
        self.sf_model = sf_model

        self.lead_time = lead_time
        self.return_gridded = return_gridded
        self.overwrite_channels = overwrite_channels
        self.base_context_exclude_tail = base_context_exclude_tail

        # TODO (nithinc): will break in torchtitan
        # Normalization buffers for broadcasted arithmetic over (B, H, W, C)
        fim = broadcast_to_4d(normalization["input_mean"])  # mean_4u_1.npy
        fis = broadcast_to_4d(normalization["input_std"])  # std_4u_1.npy
        pdm = broadcast_to_4d(normalization["pred_diff_mean"])  # mean_diff_4u_1.npy
        pds = broadcast_to_4d(normalization["pred_diff_std"])  # std_diff_4u_1.npy

        self.register_buffer("forecast_input_means", fim, persistent=False)
        self.register_buffer("forecast_input_stds", fis, persistent=False)
        self.register_buffer("forecast_pred_diff_means", pdm, persistent=False)
        self.register_buffer("forecast_pred_diff_stds", pds, persistent=False)

    def _process_se_output(
        self, forecast_y_context: torch.Tensor, se_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Non-mutating equivalent of original process_se_output.

        - forecast_y_context: (B, C, H, W) tensor
        - se_out: encoder output, typically (B, H, W, C)

        Returns:
            updated_forecast_y_context, initial_state_for_return
        """
        # Place encoder output in the first N channels of forecast context
        se_out_chw = to_channels_first_4d(se_out)  # (B, C, H, W)
        if se_out_chw.shape[1] < self.overwrite_channels:
            raise ValueError(
                f"Encoder output channels {se_out_chw.shape[1]} < overwrite_channels {self.overwrite_channels}"
            )
        if forecast_y_context.shape[1] < self.overwrite_channels:
            raise ValueError(
                f"Forecast y_context channels {forecast_y_context.shape[1]} < overwrite_channels {self.overwrite_channels}"
            )

        updated = forecast_y_context.clone()
        updated[:, : self.overwrite_channels, ...] = se_out_chw[:, : self.overwrite_channels, ...]

        # Initial state for return: de-normalised later, channels-last (B, H, W, C)
        initial_state = to_channels_last_4d(se_out)
        return updated, initial_state

    def _process_forecast_output(
        self,
        forecast_y_context: torch.Tensor,
        downscaling_y_context: torch.Tensor,
        processor_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Non-mutating equivalent of original process_forecast_output.

        Inputs:
          - forecast_y_context: (B, C, H, W)
          - downscaling_y_context: (B, C, H, W)
          - processor_out: typically (B, H, W, C)

        Returns:
          - updated_forecast_y_context (B, C, H, W)
          - updated_downscaling_y_context (B, C, H, W)
          - forecast_grid (B, H, W, C) unnormalised forecast for optional return
        """
        # Build base context from forecast y_context (channels-first -> channels-last)
        base_context = forecast_y_context[:, : -self.base_context_exclude_tail, ...]
        base_context = to_channels_last_4d(base_context)  # (B, H, W, C')

        # Un-normalise base context
        base_context_unnorm = base_context * self.forecast_input_stds + self.forecast_input_means

        # Processor produces per-channel differences; un-normalise predicted diffs
        # processor_out is already channels last format
        # x = to_channels_last_4d(processor_out)
        x = processor_out
        x = self.forecast_pred_diff_means + x * self.forecast_pred_diff_stds

        # Compute absolute forecast on channels-last
        forecast_grid = x + base_context_unnorm

        # Re-normalise for next steps
        x_norm = (forecast_grid - self.forecast_input_means) / self.forecast_input_stds  # channels-last

        # Update downscaling context first N channels with normalised forecast
        # NOTE (nithinc): I think there might be a permute missing in this logic? But I'm not sure what's actually happening
        x_norm_chw = to_channels_first_4d(x_norm)
        ds_updated = downscaling_y_context.clone()
        ds_updated[:, : self.overwrite_channels, ...] = x_norm_chw[:, : self.overwrite_channels, ...]

        # Roll forecast context: prepend new normalised channels, drop first N
        fc_updated = torch.cat([x_norm_chw, forecast_y_context[:, self.overwrite_channels :, ...]], dim=1)

        return fc_updated, ds_updated, forecast_grid

    def forward(self, task: E2ETask):
        # 1) Encode assimilation inputs
        se_out = self.se_model(task.assimilation, film_index=None)

        # 2) Seed forecast y_context with encoder output (non-mutating)
        forecast_y_context, initial_state_for_return = self._process_se_output(task.forecast.y_context, se_out)

        # Prepare a working downscaling y_context copy (non-mutating)
        downscaling_y_context = task.downscaling.y_context.clone()

        # 3) Iteratively process forecast lead times
        last_forecast_grid = None
        for lt in range(self.lead_time):
            proc_in = {"y_context": forecast_y_context, "lt": task.forecast.lt}
            proc_out = self.forecast_models[lt](proc_in, film_index=None)
            forecast_y_context, downscaling_y_context, last_forecast_grid = self._process_forecast_output(
                forecast_y_context, downscaling_y_context, proc_out
            )

        # 4) Downscaling to station predictions
        ds_task_input = {
            "y_context": downscaling_y_context,
            "x_context": [task.downscaling.x_context_lon, task.downscaling.x_context_lat],
            "x_target": task.downscaling.x_target,
            "alt_target": task.downscaling.alt_target,
            "aux_time": task.downscaling.aux_time,
            "lt": task.downscaling.lt,
        }
        station_preds = self.sf_model(ds_task_input, film_index=None)

        if self.return_gridded:
            # TODO (nithinc): potential shape bug?
            # B, W, C, H -> B, H, W, C
            initial_state_for_return = initial_state_for_return.permute(0, 3, 1, 2)
            initial_state_grid = initial_state_for_return * self.forecast_input_stds + self.forecast_input_means
            return station_preds, last_forecast_grid, initial_state_grid

        return station_preds


__all__ = [
    "ForecastTask",
    "DownscalingTask",
    "E2ETask",
    "AardvarkE2E",
]
