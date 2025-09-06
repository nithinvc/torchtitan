import pickle

import torch
import numpy as np
import torch.nn as nn

from torchtitan.experiments.weather.models.aardvark.model.layers import ConvDeepSet
from torchtitan.experiments.weather.models.aardvark.model.unet_wrap_padding import Unet
from torchtitan.experiments.weather.models.aardvark.model.vit import ViT
from torchtitan.experiments.weather.models.aardvark.model.architectures import (
    DownscalingMLP,
)

hadisd_publisher_shifts = {
    "tas": 273.15,
    "u": 0.0,
    "v": 0.0,
    "psl": 0.0,
    "ws": 0.0,
}

hadisd_publisher_scales = {
    "tas": 10,
    "u": 10,
    "v": 10,
    "psl": 100,
    "ws": 10.0,
}


def hadisd_normalisation_factors(var: str):
    path = "/home/azureuser/aux_data/norm_factors/"
    return {
        "mean": np.load(path + f"mean_hadisd_{var}.npy"),
        "std": np.load(path + f"std_hadisd_{var}.npy"),
    }


def unnormalise_hadisd_var(x, var):
    factors = hadisd_normalisation_factors(var)
    hadisd_shift = hadisd_publisher_shifts[var]
    hadisd_scale = hadisd_publisher_scales[var]

    return hadisd_shift + hadisd_scale * (factors["mean"] + factors["std"] * x)


class ConvCNPWeatherOnToOff(nn.Module):
    """
    ConvCNP for decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        int_channels,
        device,
        res,
        data_path="../data/",
        mode="end_to_end",
        decoder=None,
        film=False,
    ):
        super().__init__()

        # Setup
        self.device = device

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder
        self.int_x = 256
        self.int_y = 128
        self.mode = mode
        self.film = film

        # Load lon-lat of internal discretisation
        self.era5_x = (
            torch.from_numpy(
                np.load(data_path + "grid_lon_lat/era5_x_{}.npy".format(res))
            ).float()
            / 360
        )
        self.era5_y = (
            torch.from_numpy(
                np.load(data_path + "grid_lon_lat/era5_y_{}.npy".format(res))
            ).float()
            / 360
        )

        # Setup setconv
        self.sc_out = ConvDeepSet(
            0.001, "OnToOff", density_channel=False, device=self.device
        )

        if self.mode not in ["downscaling", "end_to_end"]:
            unet_out_channels = out_channels
        else:
            unet_out_channels = int_channels

        # UNet backbone
        if self.decoder == "base":
            self.decoder_lr = Unet(
                in_channels=in_channels,
                out_channels=unet_out_channels,
                div_factor=1,
                film=film,
            )

        else:
            raise Exception(f"Expected to use base decoder, but got {self.decoder}")

        # Postprocessing MLP
        self.mlp = DownscalingMLP(
            in_channels=24 + 9,
            out_channels=1,
            h_channels=64,
            h_layers=2,
        )
        self.proj = nn.Linear(27, 33)

    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, task, film_index):
        x = task["y_context"]
        batch_size = x.shape[0]

        # UNet backbone
        lt = torch.zeros((batch_size, 1)).to(x.device)
        x = self.decoder_lr(x, film_index=lt)

        # Transform to station predictions with setconv
        num_channels = x.shape[3]
        x = x.permute(0, 3, 1, 2)
        assert list(x.shape) == [batch_size, num_channels, 240, 121]
        x_target = task["x_target"]
        num_stations = x_target.shape[2]

        x = self.sc_out(
            x_in=task["x_context"],
            wt=x,
            x_out=[x_target[:, 0, :], x_target[:, 1, :]],
        )
        assert x.shape[0] == batch_size
        assert x.shape[2] == num_stations

        # Concatenate auxiliary data at stations
        # alt_target = task["alt_target"]
        # assert torch.isnan(alt_target).sum() == 0
        # assert alt_target.shape[0] == batch_size
        # assert alt_target.shape[2] == num_stations

        aux_time = torch.zeros((batch_size, 1, num_stations)).cuda().float()
        # aux_time = task["aux_time"].squeeze(-1).repeat(1, 1, num_stations)

        assert aux_time.shape[0] == batch_size
        assert aux_time.shape[2] == num_stations

        # x = torch.cat([x, alt_target, x_target, aux_time], dim=1).permute(0, 2, 1)
        x = torch.cat([x, x_target, aux_time], dim=1).permute(0, 2, 1)
        assert x.shape[0] == batch_size
        assert x.shape[1] == num_stations

        x = self.proj(x)

        tmp = self.mlp(x)
        assert list(tmp.shape) == [batch_size, num_stations, 1]
        y_hat = tmp.squeeze(-1)
        assert list(y_hat.shape) == [batch_size, num_stations]
        return y_hat
