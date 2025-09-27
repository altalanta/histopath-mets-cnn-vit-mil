"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """Snapshot of RNG state for reproducibility checks."""

    python_state: tuple[object, ...]
    numpy_state: tuple[object, ...]
    torch_state: tuple[object, ...]


def seed_everything(seed: int) -> SeedState:
    """Seed Python, NumPy, and PyTorch for deterministic runs.

    Returns the RNG state so callers can restore if needed.
    """

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    return SeedState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state().clone(),
    )


def restore_seed_state(state: SeedState) -> None:
    """Restore RNG state captured by :func:`seed_everything`."""

    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state)


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Factory for deterministic DataLoader worker initialisation."""

    def _init(worker_id: int) -> None:
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return _init
