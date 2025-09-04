import torch
import torch.nn as nn


class ConvDeepSet(nn.Module):
    """
    Simplified convDeepSet equivalent extracted from working_dir/set_convs.py
    Supports OffToOn, OnToOn, and OnToOff with density channel handling.
    """

    def __init__(self, init_ls: float, mode: str, device: str, density_channel: bool = True):
        super().__init__()
        self.init_ls = torch.nn.Parameter(torch.tensor([init_ls]))
        self.mode = mode
        self.density_channel = density_channel
        self.init_ls.requires_grad = True
        self.device = device

    def compute_weights(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dists2 = self.pw_dists2(x1.unsqueeze(-1), x2.unsqueeze(-1))
        return torch.exp((-0.5 * dists2) / (self.init_ls.to(x1.device)).pow(2))

    def pw_dists2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        norms_a = torch.sum(a.pow(2), dim=-1)[..., :, None]
        norms_b = torch.sum(b.pow(2), dim=-1)[..., None, :]
        return norms_a + norms_b - 2 * torch.matmul(a, b.permute(0, 2, 1))

    def forward(self, x_in, wt: torch.Tensor, x_out):
        density_channel = torch.ones_like(wt[:, 0:1, ...])
        density_channel[torch.isnan(wt[:, 0:1, ...])] = 0
        wt = torch.cat([density_channel, wt], dim=1)
        wt[torch.isnan(wt)] = 0

        if self.mode == "OffToOn":
            in_lon_mask = ~torch.isnan(x_in[0])
            in_lat_mask = ~torch.isnan(x_in[1])
            x_in[0][~in_lon_mask] = 0
            x_in[1][~in_lat_mask] = 0
            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]
            ws[0] = ws[0] * in_lon_mask.unsqueeze(-1).int()
            ws[1] = ws[1] * in_lat_mask.unsqueeze(-1).int()
            ee = torch.einsum("...cw,...wx,...wy->...cxy", wt, ws[0], ws[1])

        elif self.mode == "OnToOn":
            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]
            ee = torch.einsum("...cwh,...wx,...hy->...cxy", wt, ws[0], ws[1])

        elif self.mode == "OnToOff":
            out_lon_mask = ~torch.isnan(x_out[0])
            out_lat_mask = ~torch.isnan(x_out[1])
            x_out[0][~out_lon_mask] = 0
            x_out[1][~out_lat_mask] = 0
            ws = [self.compute_weights(xzi, xi) for xzi, xi in zip(x_in, x_out)]
            ws[0] = ws[0] * out_lon_mask.unsqueeze(-2).int()
            ws[1] = ws[1] * out_lat_mask.unsqueeze(-2).int()
            ee = torch.einsum("...cwh,...wx,...hx->...cx", wt, ws[0], ws[1])
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if self.density_channel:
            ee = torch.cat(
                [ee[:, 0:1, ...], ee[:, 1:, ...] / torch.clamp(ee[:, 0:1, ...], min=1e-6, max=1e5)],
                dim=1,
            )
            return ee
        else:
            ee = ee[:, 1:, ...] / torch.clamp(ee[:, 0:1, ...], min=1e-6, max=1e5)
            return ee


__all__ = [
    "ConvDeepSet",
]


