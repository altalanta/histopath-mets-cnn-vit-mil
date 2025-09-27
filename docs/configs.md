# Configuration

Configs live in `configs/core.yaml` and use Hydra for composition.

## Modes

- `mode=fast_debug` (default): small synthetic dataset, 1–2 epochs, CPU-only dataloaders.
- `mode=full`: large dataset, deeper encoders, higher batch sizes for GPU.

Switch modes via CLI overrides:

```bash
histo-omics-lite train simclr --override mode=full
```

## Paths

`paths.synthetic_root`, `paths.runs_root`, and `paths.reports_root` control where data, checkpoints, and reports land. Override them per experiment:

```bash
histo-omics-lite train clip \
  --override paths.synthetic_root=/mnt/shards \
  --override paths.runs_root=/mnt/runs
```

## Profiling

Set `profiling.enabled=true` to capture Chrome traces in each run directory (see `utils/profiler.py`). Activities default to CPU but can include CUDA when available.
