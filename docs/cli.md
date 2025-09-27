# CLI Reference

All commands accept repeated `--override key=value` to pass Hydra overrides and a global `--seed` for determinism.

## `data make-synthetic`

- `--train`, `--val`: number of samples per split.
- `--out`: destination directory (default `data/synthetic`).
- `--device`: `auto|cpu|cuda` (used if you generate data with torch ops on GPU).

Example:

```bash
histo-omics-lite data make-synthetic \
  --train 2048 --val 512 \
  --out /data/histo-lite \
  --override data.synthetic.omics.dim=128
```

## `train simclr`

- Outputs best checkpoint to `<out>/simclr_best.pt`.
- Logs JSON history + optional profiler trace.

```bash
histo-omics-lite train simclr --out runs/simclr --override training.simclr.epochs=5
```

## `train clip`

- Optional `--simclr-checkpoint` to warm-start the image backbone.
- Saves `<out>/clip_best.pt` and training history.

```bash
histo-omics-lite train clip \
  --simclr-checkpoint runs/simclr/simclr_best.pt \
  --override training.clip.batch_size=64
```

## `eval retrieval`

- Emits `metrics.json`, `umap.png`, `gradcam_*.png`, and `embeddings.json` into the output directory.
- Accepts `--data` to override the dataset root (otherwise uses config paths).

## `infer embed`

Batch exports embeddings to Parquet (default) or JSON based on the output suffix.

```bash
histo-omics-lite infer embed data/synthetic \
  --checkpoint runs/clip/clip_best.pt \
  --out embeddings.parquet
```
