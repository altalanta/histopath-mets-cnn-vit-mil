# Bring Your Own Data

## Step 1 – Prepare tiles

- Tile WSIs into PNGs or JPEGs (RGB, 8-bit) at a consistent magnification.
- Store tiles inside WebDataset shards with key `tile.png` and include a JSON sidecar per sample.

## Step 2 – Prepare omics tensors

- Assemble transcriptomics features into float32 vectors.
- Apply per-gene normalisation (log-transform + z-score recommended).
- Save vectors as `.npy` inside the same sample record.

## Step 3 – Build metadata tables

- Create `samples.parquet` with columns `sample_id`, `split`, `label`, `label_name`, `shard`, `key`.
- Use Polars to enforce the schema from `histo_omics_lite/data/schemas.py`.

## Step 4 – Update config overrides

```bash
histo-omics-lite train clip   --override paths.synthetic_root=/data/my_shards   --override data.synthetic.omics.dim=512   --override training.clip.batch_size=64
```

## Step 5 – Scale carefully

- Increase `mode` gradually (`fast_debug` -> custom settings -> `full`).
- Monitor throughput logs; adjust `num_workers` and `batch_size` to balance CPU load and memory.
- Capture profiler traces on the first few runs to check for dataloader bottlenecks.
