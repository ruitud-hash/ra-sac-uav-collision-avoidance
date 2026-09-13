"""Helpers for transferring scalar training metrics to the CPU."""

from __future__ import annotations

import torch


def scalar_values(*values: torch.Tensor) -> tuple[float, ...]:
    """Transfer scalar tensors to the CPU in one synchronization."""
    packed = torch.stack([value.detach().reshape(()) for value in values])
    return tuple(float(value) for value in packed.cpu().tolist())
