# Histo-Omics Lite

**histo-omics-lite** is a laptop-first toolkit for contrastive learning on histology tiles and transcriptomics vectors. It provides:

- Deterministic SimCLR pretraining of a ResNet-18 tile encoder.
- CLIP-style multimodal alignment between histology and omics signatures.
- A reproducible WebDataset pipeline with schema validation and synthetic data generators.
- Batteries-included evaluation with retrieval metrics, confidence intervals, UMAP projections, and Grad-CAM overlays.
- A single Typer-based CLI that keeps configuration in Hydra and surfaces overrides when you need them.

The project is designed for rapid experimentation on CPU hardware while remaining CUDA-ready when you need acceleration.
