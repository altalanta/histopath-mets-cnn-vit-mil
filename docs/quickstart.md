# Quickstart

The default configuration targets CPU-only machines and completes the end-to-end smoke test in well under a minute.

```bash
# 1. Create synthetic shards and tables
histo-omics-lite data make-synthetic --train 256 --val 64 --out data/synthetic

# 2. SimCLR pretraining (fast_debug mode is the default)
histo-omics-lite train simclr --out runs/simclr

# 3. CLIP-style multimodal alignment
histo-omics-lite train clip --out runs/clip --simclr-checkpoint runs/simclr/simclr_best.pt

# 4. Retrieval evaluation with figures + metrics
histo-omics-lite eval retrieval \
  --checkpoint runs/clip/clip_best.pt \
  --data data/synthetic \
  --out reports/retrieval

# 5. Embed new tiles/omics in batch
histo-omics-lite infer embed data/synthetic --checkpoint runs/clip/clip_best.pt --out embeddings.parquet
```

All commands accept Hydra overrides via repeated `--override key=value` flags. For a longer run on a GPU, switch to the full profile:

```bash
histo-omics-lite train simclr --override mode=full
```
