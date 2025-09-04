import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(n_channels, n_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class DownscalingMLP(nn.Module):
    """
    MLP for handling auxiliary data at station locations
    Mirrors the structure used in working_dir with residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, h_channels: int, h_layers: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            *[ResidualBlock(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLP(nn.Module):
    """
    Simple MLP used by ViT when per_var_embedding=False (mirrors working_dir).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        h_channels: int = 64,
        h_layers: int = 4,
    ):
        super().__init__()

        def hidden_block(hc: int):
            return nn.Sequential(nn.Linear(hc, hc), nn.ReLU())

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            nn.ReLU(),
            *[hidden_block(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


__all__ = [
    "ResidualBlock",
    "DownscalingMLP",
    "MLP",
]
