"""Device selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved device settings for compute and tensors."""

    device: torch.device
    use_cuda: bool


def resolve_device(choice: str = "auto") -> DeviceConfig:
    """Resolve the desired device based on the user's override string."""

    normalized = choice.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device choice: {choice}")

    if normalized == "cpu":
        device = torch.device("cpu")
        return DeviceConfig(device=device, use_cuda=False)

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return DeviceConfig(device=torch.device("cuda"), use_cuda=True)

    if torch.cuda.is_available():
        return DeviceConfig(device=torch.device("cuda"), use_cuda=True)

    return DeviceConfig(device=torch.device("cpu"), use_cuda=False)
