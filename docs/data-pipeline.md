# Data Pipeline

## Synthetic shards

- `histo-omics-lite data make-synthetic` fabricates histology tiles and paired omics vectors.
- WebDataset shards contain `PNG` tiles, `NPY` expression vectors, and JSON metadata.
- Every run validates records against a Pydantic schema and enforces a typed Polars table.
- Manifest files summarise class balance, image size, transcript dimension, and shard layout.

```bash
histo-omics-lite data make-synthetic \
  --train 1024 --val 256 \
  --out data/synthetic \
  --override mode=fast_debug
```

## Bringing external data

To work with your own tiles and omics:

1. Produce WebDataset shards where each sample includes `tile.png`, `transcript.npy`, and `sample.json` with keys `sample_id`, `split`, `label`, and `label_name`.
2. Generate a Polars table (Parquet) mirroring `tables/samples.parquet`. Use the schema in `histo_omics_lite/data/schemas.py` as reference.
3. Point the pipeline at your shards by overriding `paths.synthetic_root` (for now) and the omics dimensionality if it differs.

## Omics normalisation

The synthetic generator produces mean-zero, unit-variance features per gene. For real data, normalise counts to a comparable scale before alignment (e.g. log(TPM+1) followed by z-scoring per gene). Mismatched scales will destabilise the CLIP temperature and degrade retrieval performance.
