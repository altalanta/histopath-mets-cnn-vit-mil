"""Test configuration and fixtures for pytest."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Provide a device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image():
    """Provide a sample image for testing."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_images():
    """Provide multiple sample images for testing."""
    return [
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        for _ in range(5)
    ]


@pytest.fixture
def sample_features():
    """Provide sample feature tensors for MIL testing."""
    return torch.randn(4, 20, 512)


@pytest.fixture
def sample_batch():
    """Provide a sample batch for testing."""
    return {
        "image": torch.randn(2, 3, 224, 224),
        "label": torch.tensor([0, 1]),
        "slide_id": ["slide_001", "slide_002"]
    }


@pytest.fixture
def sample_mil_batch():
    """Provide a sample MIL batch for testing."""
    return {
        "images": torch.randn(2, 10, 3, 224, 224),
        "label": torch.tensor([0, 1]),
        "bag_id": ["bag_001", "bag_002"]
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "tiles").mkdir(parents=True)
    
    return data_dir


@pytest.fixture(scope="session", autouse=True)
def set_torch_threads():
    """Set number of torch threads for testing."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


@pytest.fixture(autouse=True)
def seed_random():
    """Seed random number generators for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)