#!/usr/bin/env python3
"""Update performance baseline for CI budget tests."""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.test_budget import run_dataloader_benchmark


def update_baseline():
    """Update the performance baseline."""
    print("Running comprehensive benchmark to establish new baseline...")
    
    # Run multiple configurations and take median
    results = []
    
    configurations = [
        {"n_samples": 100, "batch_size": 16, "n_epochs": 5},
        {"n_samples": 100, "batch_size": 32, "n_epochs": 5},
        {"n_samples": 200, "batch_size": 16, "n_epochs": 3},
    ]
    
    for config in configurations:
        print(f"Testing config: {config}")
        perf = run_dataloader_benchmark(**config, num_workers=0, seed=42)
        print(f"  Performance: {perf:.2f} steps/sec")
        results.append(perf)
    
    # Use median performance as baseline
    import numpy as np
    baseline_perf = np.median(results)
    
    print(f"\nNew baseline performance: {baseline_perf:.2f} steps/sec")
    print(f"Range: {min(results):.2f} - {max(results):.2f} steps/sec")
    
    # Create baseline data
    baseline_data = {
        "steps_per_sec": float(baseline_perf),
        "last_updated": datetime.now().isoformat(),
        "hardware": f"{platform.system()} {platform.machine()}",
        "python_version": platform.python_version(),
        "configurations_tested": configurations,
        "all_results": results,
        "notes": "Updated via update_baseline.py"
    }
    
    # Save baseline
    baseline_path = Path(__file__).parent.parent / "benchmarks" / "baseline.json"
    baseline_path.parent.mkdir(exist_ok=True)
    
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"\nBaseline saved to: {baseline_path}")
    print("\nYou can now run: pytest tests/test_budget.py")


if __name__ == "__main__":
    update_baseline()