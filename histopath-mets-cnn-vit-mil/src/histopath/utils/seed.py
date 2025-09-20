"""Reproducibility utilities for seeding and deterministic training."""

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all random number generators for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if deterministic:
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Allow non-deterministic for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a PyTorch generator with optional seed.
    
    Args:
        seed: Random seed for the generator
        
    Returns:
        PyTorch generator instance
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """
    Initialize worker processes for DataLoader with different seeds.
    
    Args:
        worker_id: Worker process ID
        base_seed: Base seed value
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)