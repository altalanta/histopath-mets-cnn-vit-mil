# Troubleshooting

## WebDataset ingest stalls

- Ensure shards are readable: `wds.utils.peek("data/synthetic/shards/train/*.tar")`.
- Validate metadata table: `polars.read_parquet("tables/samples.parquet").describe()`.
- If you mutate shards externally, regenerate the manifest to keep hashes in sync.

## CUDA alignment diverges

- Check that your PyTorch wheel matches the installed CUDA toolkit (see install guide).
- Use `--override training.clip.log_interval=10` to surface loss spikes early.
- Enable profiling with `--override profiling.enabled=true` and inspect `profiler_trace.json`.

## Determinism checks

- Always set `--seed` when invoking the CLI for reproducible runs.
- `histo-omics-lite eval retrieval --seed 42` guarantees identical metrics and plots.

## Missing Grad-CAM figures

- Ensure you installed the `viz` extra so Matplotlib and seaborn are available.
- If running headless, set `MPLBACKEND=Agg` before invoking the CLI.
