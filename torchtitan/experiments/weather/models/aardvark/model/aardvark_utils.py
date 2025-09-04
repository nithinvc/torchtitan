import torch


def to_channels_last_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is (B, H, W, C).
    Accepted inputs:
      - (B, H, W, C)
      - (B, C, H, W) -> permutes to (B, H, W, C)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1)


def to_channels_first_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is (B, C, H, W).
    Accepted inputs:
      - (B, C, H, W)
      - (B, H, W, C) -> permutes to (B, C, H, W)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
    return x.permute(0, 3, 1, 2)


def broadcast_to_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Normalization factors can be provided as (C,) or (1, 1, 1, C).
    Return value is (1, 1, 1, C) for broadcasting over (B, H, W, C).
    """
    if x.dim() == 1:
        return x.view(1, 1, 1, -1)
    if x.dim() == 4:
        if x.shape[0] == 1 and x.shape[1] == 1 and x.shape[2] == 1:
            return x
    raise ValueError(f"Normalization tensor must be (C,) or (1,1,1,C), got shape {tuple(x.shape)}")


__all__ = [
    "to_channels_last_4d",
    "to_channels_first_4d",
    "broadcast_to_4d",
]
