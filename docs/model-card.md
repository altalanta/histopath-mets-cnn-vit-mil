# Model Card

## Model details

- **Encoders**: ResNet-18 backbone with two-layer projection head for histology; two-layer MLP for omics.
- **Pretraining**: SimCLR with NT-Xent loss, deterministic augmentations tuned for tissue tiles.
- **Alignment**: Symmetric CLIP loss with temperature parameterised in log space.

## Intended use

- Rapid experimentation on synthetic or lightweight histology + omics datasets.
- Benchmarking retrieval quality and interpretability (UMAP, Grad-CAM) before scaling to full cohorts.

## Limitations

- Synthetic generator is not a substitute for domain-specific tissue morphology.
- Omics vectors assume per-gene normalisation to zero mean and unit variance.
- Retrieval metrics are computed on small validation splits; confidence intervals may be wide.

## Ethical considerations

- No real patient data is distributed with this project.
- When transferring to clinical data, ensure proper de-identification and IRB approval.
