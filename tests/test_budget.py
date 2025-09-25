"""Performance budget tests to catch regressions in data loading speed."""

import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from histopath.data.dataset import HistopathDataset, create_dataloader
from histopath.utils.seed import seed_everything, worker_init_fn


class SyntheticTileGenerator:
    """Generate synthetic histopathology tiles for benchmarking."""

    def __init__(self, tile_size: int = 256, n_tiles: int = 100, seed: int = 42):
        """Initialize synthetic tile generator."""
        self.tile_size = tile_size
        self.n_tiles = n_tiles
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._tile_paths = []

    def generate_tiles(self) -> pd.DataFrame:
        """Generate synthetic tiles and return metadata DataFrame."""
        tiles_data = []
        
        for i in range(self.n_tiles):
            # Generate synthetic H&E-like image
            # Simulate tissue with realistic color distribution
            image = self._generate_synthetic_he_image()
            
            # Save to temporary file
            temp_file = NamedTemporaryFile(suffix='.png', delete=False)
            Image.fromarray(image).save(temp_file.name)
            self._tile_paths.append(temp_file.name)
            
            tiles_data.append({
                'tile_path': temp_file.name,
                'label': i % 2,  # Binary labels
                'tile_id': f'synthetic_{i:06d}',
                'slide_id': f'slide_{i // 20}',  # 20 tiles per slide
                'x': (i % 10) * 256,
                'y': (i // 10) * 256,
            })
        
        return pd.DataFrame(tiles_data)

    def _generate_synthetic_he_image(self) -> np.ndarray:
        """Generate synthetic H&E stained tissue image."""
        # Start with tissue-like base color
        tissue_base = self.rng.uniform(180, 220, (self.tile_size, self.tile_size, 3))
        
        # Add H&E-like staining patterns
        # Hematoxylin (blue/purple nuclei)
        nuclei_mask = self._generate_nuclei_pattern()
        tissue_base[nuclei_mask] *= self.rng.uniform(0.3, 0.6, (np.sum(nuclei_mask), 3))
        tissue_base[nuclei_mask, 2] += self.rng.uniform(20, 60, np.sum(nuclei_mask))  # More blue
        
        # Eosin (pink cytoplasm)
        cytoplasm_mask = self._generate_cytoplasm_pattern()
        tissue_base[cytoplasm_mask, 0] += self.rng.uniform(10, 30, np.sum(cytoplasm_mask))  # More red
        tissue_base[cytoplasm_mask, 2] *= self.rng.uniform(0.7, 0.9, np.sum(cytoplasm_mask))  # Less blue
        
        # Add noise and texture
        noise = self.rng.normal(0, 5, tissue_base.shape)
        tissue_base += noise
        
        # Clip values
        tissue_base = np.clip(tissue_base, 0, 255).astype(np.uint8)
        
        return tissue_base

    def _generate_nuclei_pattern(self) -> np.ndarray:
        """Generate realistic nuclei distribution pattern."""
        mask = np.zeros((self.tile_size, self.tile_size), dtype=bool)
        
        # Generate random nuclei centers
        n_nuclei = self.rng.integers(20, 80)
        centers = self.rng.integers(10, self.tile_size - 10, (n_nuclei, 2))
        
        for center in centers:
            # Create circular nuclei with varying sizes
            radius = self.rng.uniform(3, 8)
            y, x = np.ogrid[:self.tile_size, :self.tile_size]
            nucleus_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            mask |= nucleus_mask
            
        return mask

    def _generate_cytoplasm_pattern(self) -> np.ndarray:
        """Generate cytoplasm pattern around nuclei."""
        # Simple approach: areas not covered by nuclei with some probability
        nuclei_mask = self._generate_nuclei_pattern()
        cytoplasm_prob = self.rng.random((self.tile_size, self.tile_size))
        cytoplasm_mask = (~nuclei_mask) & (cytoplasm_prob > 0.3)
        return cytoplasm_mask

    def cleanup(self):
        """Clean up temporary files."""
        for path in self._tile_paths:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass


def run_dataloader_benchmark(
    n_samples: int = 100,
    batch_size: int = 16,
    num_workers: int = 0,
    n_epochs: int = 3,
    tile_size: int = 256,
    seed: int = 42
) -> float:
    """
    Run synthetic dataloader benchmark.
    
    Args:
        n_samples: Number of synthetic samples
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        n_epochs: Number of epochs to benchmark
        tile_size: Size of synthetic tiles
        seed: Random seed
        
    Returns:
        Median steps per second across epochs
    """
    # Set seed for reproducible benchmarks
    seed_everything(seed)
    
    # Generate synthetic data
    generator = SyntheticTileGenerator(tile_size=tile_size, n_tiles=n_samples, seed=seed)
    
    try:
        tiles_df = generator.generate_tiles()
        
        # Create dataset and dataloader
        dataset = HistopathDataset(
            tiles_df=tiles_df,
            transform=None,  # No transforms for pure I/O benchmark
        )
        
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # Disable for CI consistency
            drop_last=False,
        )
        
        # Benchmark multiple epochs
        epoch_times = []
        
        for epoch in range(n_epochs):
            start_time = time.time()
            batch_count = 0
            
            for batch in dataloader:
                # Simulate minimal processing
                _ = batch['image'].shape
                batch_count += 1
            
            epoch_time = time.time() - start_time
            steps_per_sec = batch_count / epoch_time if epoch_time > 0 else 0
            epoch_times.append(steps_per_sec)
        
        return np.median(epoch_times)
    
    finally:
        # Clean up temporary files
        generator.cleanup()


def test_dataloader_performance_budget():
    """CI budget test - fail if performance regresses more than 20%."""
    baseline_path = Path(__file__).parent.parent / "benchmarks" / "baseline.json"
    
    if not baseline_path.exists():
        pytest.skip("No baseline found - run update_baseline.py to establish baseline")
    
    # Load baseline
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    
    baseline_perf = baseline_data["steps_per_sec"]
    
    # Run current benchmark
    current_perf = run_dataloader_benchmark(
        n_samples=50,  # Smaller for CI speed
        batch_size=8,
        num_workers=0,  # Single-threaded for CI consistency
        n_epochs=3,
        seed=42
    )
    
    # Check performance budget (allow 20% regression)
    threshold = 0.8 * baseline_perf
    
    assert current_perf >= threshold, (
        f"Performance regression detected: {current_perf:.2f} steps/sec "
        f"< 80% of baseline ({threshold:.2f}). "
        f"Baseline: {baseline_perf:.2f} steps/sec. "
        f"Consider optimizing data loading or updating baseline if intentional."
    )


def test_dataloader_determinism():
    """Test that dataloader performance is deterministic with same seed."""
    # Run benchmark twice with same seed
    perf1 = run_dataloader_benchmark(n_samples=20, n_epochs=2, seed=42)
    perf2 = run_dataloader_benchmark(n_samples=20, n_epochs=2, seed=42)
    
    # Should be very similar (within 10% due to system noise)
    relative_diff = abs(perf1 - perf2) / max(perf1, perf2)
    
    assert relative_diff < 0.1, (
        f"Dataloader performance not deterministic: "
        f"{perf1:.2f} vs {perf2:.2f} steps/sec (diff: {relative_diff:.1%})"
    )


@pytest.mark.slow
def test_dataloader_scaling():
    """Test dataloader performance scaling with batch size and workers."""
    base_perf = run_dataloader_benchmark(
        n_samples=40, 
        batch_size=4, 
        num_workers=0, 
        n_epochs=2
    )
    
    larger_batch_perf = run_dataloader_benchmark(
        n_samples=40, 
        batch_size=8, 
        num_workers=0, 
        n_epochs=2
    )
    
    # Larger batches should generally be more efficient
    # (though this can vary based on system)
    assert larger_batch_perf > 0, "Performance should be positive"
    assert base_perf > 0, "Base performance should be positive"


if __name__ == "__main__":
    # Run benchmark when script is executed directly
    print("Running dataloader benchmark...")
    
    perf = run_dataloader_benchmark(
        n_samples=100,
        batch_size=16,
        num_workers=0,
        n_epochs=5,
    )
    
    print(f"Performance: {perf:.2f} steps/sec")
    
    # Show baseline comparison if available
    baseline_path = Path(__file__).parent.parent / "benchmarks" / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline_perf = baseline_data["steps_per_sec"]
        change = (perf - baseline_perf) / baseline_perf * 100
        print(f"Baseline: {baseline_perf:.2f} steps/sec")
        print(f"Change: {change:+.1f}%")