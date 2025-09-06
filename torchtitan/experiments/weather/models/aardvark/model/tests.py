import unittest
from typing import Any, cast

import torch
import torch.nn as nn

from tensordict import TensorDict  # type: ignore

from torchtitan.experiments.weather.models.aardvark.model.model import (
    AardvarkE2E,
    ForecastTask,
    DownscalingTask,
    E2ETask,
)


class DummyEncoder(nn.Module):
    def forward(self, assimilation, film_index=None):
        return assimilation["enc"]


class DummyProcessor(nn.Module):
    def __init__(self, delta_value: float, out_channels: int):
        super().__init__()
        self.delta_value = delta_value
        self.out_channels = out_channels

    def forward(self, task, film_index=None):
        y = task["y_context"]
        b, c, h, w = y.shape
        return torch.full(
            (b, h, w, self.out_channels),
            self.delta_value,
            dtype=y.dtype,
            device=y.device,
        )


class DummyDecoder(nn.Module):
    def forward(self, task, film_index=None):
        # Simple, deterministic mapping: mean across H,W of first channel, replicated per station
        y = task["y_context"]  # (B, C, H, W)
        b, c, h, w = y.shape
        num_stations = task["x_target"].shape[-1]
        mean0 = y[:, 0:1].mean(dim=(-1, -2))  # (B,1)
        return mean0.repeat(1, num_stations)


def ref_process(
    overwrite_channels: int,
    base_context_exclude_tail: int,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    pred_diff_mean: torch.Tensor,
    pred_diff_std: torch.Tensor,
    se_out: torch.Tensor,
    forecast_y_context: torch.Tensor,
    downscaling_y_context: torch.Tensor,
    processors_out: list,
):
    # Seed forecast
    se_out_chw = se_out.permute(0, 3, 1, 2)
    fc = forecast_y_context.clone()
    fc[:, :overwrite_channels] = se_out_chw[:, :overwrite_channels]
    initial_state = se_out

    fim = input_mean.view(1, 1, 1, -1)
    fis = input_std.view(1, 1, 1, -1)
    pdm = pred_diff_mean.view(1, 1, 1, -1)
    pds = pred_diff_std.view(1, 1, 1, -1)

    ds = downscaling_y_context.clone()
    last_forecast = None

    for p_out in processors_out:
        base = fc[:, :-base_context_exclude_tail]  # (B,C',H,W)
        base = base.permute(0, 2, 3, 1)  # (B,H,W,C')
        base_unnorm = base * fis + fim

        x = p_out.permute(0, 2, 3, 1) if p_out.shape[1] > p_out.shape[-1] else p_out
        x = pdm + x * pds

        forecast = x + base_unnorm
        last_forecast = forecast
        x_norm = (forecast - fim) / fis
        x_norm_chw = x_norm.permute(0, 3, 1, 2)

        ds[:, :overwrite_channels] = x_norm_chw[:, :overwrite_channels]
        fc = torch.cat([x_norm_chw, fc[:, overwrite_channels:]], dim=1)

    return fc, ds, last_forecast, initial_state


class TestAardvarkE2E(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 2
        self.H = 4
        self.W = 3
        self.C = 24
        self.C_tail = 11
        self.C_total = self.C + self.C_tail
        self.Ns = 5

        self.input_mean = torch.zeros(self.C)
        self.input_std = torch.ones(self.C)
        self.pred_diff_mean = torch.zeros(self.C)
        self.pred_diff_std = torch.ones(self.C)

        self.overwrite_channels = self.C
        self.base_context_exclude_tail = self.C_tail

        # Build stubs
        self.encoder = DummyEncoder()
        self.forecasters = [
            DummyProcessor(delta_value=0.1, out_channels=self.C) for _ in range(3)
        ]
        self.decoder = DummyDecoder()

        self.model = AardvarkE2E(
            se_model=self.encoder,
            forecast_models=self.forecasters,
            sf_model=self.decoder,
            lead_time=len(self.forecasters),
            normalization={
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "pred_diff_mean": self.pred_diff_mean,
                "pred_diff_std": self.pred_diff_std,
            },
            return_gridded=False,
            overwrite_channels=self.overwrite_channels,
            base_context_exclude_tail=self.base_context_exclude_tail,
        )

        # Build task
        enc = torch.randn(self.B, self.H, self.W, self.C)
        self.assimilation = TensorDict({"enc": enc}, batch_size=[self.B])

        # Forecast and downscaling contexts include extra tail channels as in original code
        y_context_fc = torch.zeros(self.B, self.C_total, self.H, self.W)
        lt = torch.ones(self.B, 1)
        self.forecast = cast(Any, ForecastTask)(y_context=y_context_fc, lt=lt)

        y_context_ds = torch.zeros(self.B, self.C_total, self.H, self.W)
        x_context_lon = torch.zeros(self.B, 240)  # not used by dummy
        x_context_lat = torch.zeros(self.B, 121)  # not used by dummy
        x_target = torch.zeros(self.B, 2, self.Ns)
        alt_target = torch.zeros(self.B, 1, self.Ns)
        aux_time = torch.zeros(self.B, 1, 1)
        self.downscaling = cast(Any, DownscalingTask)(
            y_context=y_context_ds,
            x_context_lon=x_context_lon,
            x_context_lat=x_context_lat,
            x_target=x_target,
            alt_target=alt_target,
            aux_time=aux_time,
            lt=lt,
        )

        self.task = cast(Any, E2ETask)(
            assimilation=self.assimilation,
            forecast=self.forecast,
            downscaling=self.downscaling,
        )

    def test_equivalence_reference_logic(self):
        # Run model
        out = self.model(self.task)

        # Reference computation using same stubs and math
        # Processors_out are channel-last diffs
        processors_out = [
            torch.full((self.B, self.H, self.W, self.C), 0.1)
            for _ in range(len(self.forecasters))
        ]

        fc, ds, last_forecast, initial_state = ref_process(
            self.overwrite_channels,
            self.base_context_exclude_tail,
            self.input_mean,
            self.input_std,
            self.pred_diff_mean,
            self.pred_diff_std,
            se_out=self.assimilation["enc"],
            forecast_y_context=self.forecast.y_context,
            downscaling_y_context=self.downscaling.y_context,
            processors_out=processors_out,
        )

        # Decoder on reference
        ref_task = {
            "y_context": ds,
            "x_context": [
                self.downscaling.x_context_lon,
                self.downscaling.x_context_lat,
            ],
            "x_target": self.downscaling.x_target,
            "alt_target": self.downscaling.alt_target,
            "aux_time": self.downscaling.aux_time,
            "lt": self.downscaling.lt,
        }
        ref_out = self.decoder(ref_task)

        self.assertTrue(torch.allclose(out, ref_out, atol=1e-6))

    def test_non_mutation(self):
        # Keep originals
        fc0 = self.task.forecast.y_context.clone()
        ds0 = self.task.downscaling.y_context.clone()

        _ = self.model(self.task)

        self.assertTrue(torch.equal(fc0, self.task.forecast.y_context))
        self.assertTrue(torch.equal(ds0, self.task.downscaling.y_context))

    def test_return_gridded(self):
        model = AardvarkE2E(
            se_model=self.encoder,
            forecast_models=self.forecasters,
            sf_model=self.decoder,
            lead_time=len(self.forecasters),
            normalization={
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "pred_diff_mean": self.pred_diff_mean,
                "pred_diff_std": self.pred_diff_std,
            },
            return_gridded=True,
            overwrite_channels=self.overwrite_channels,
            base_context_exclude_tail=self.base_context_exclude_tail,
        )

        y, forecast_grid, initial_grid = model(self.task)

        self.assertEqual(y.shape, (self.B, self.Ns))
        self.assertEqual(forecast_grid.shape, (self.B, self.H, self.W, self.C))
        self.assertEqual(initial_grid.shape, (self.B, self.H, self.W, self.C))


if __name__ == "__main__":
    unittest.main()
