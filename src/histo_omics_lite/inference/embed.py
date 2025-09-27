"""Embedding export routines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl
import torch
from rich.console import Console

from ..data.pipeline import build_embedding_loader
from ..data.transforms import ClipTransformConfig, build_clip_image_transform
from ..training.models import ClipAlignmentModel, ProjectionConfig
from ..utils.config import load_config, to_dict
from ..utils.device import resolve_device
from ..utils.seed import seed_everything

console = Console()


def _device_choice(config_device: str, override: str) -> str:
    if override != "auto":
        return override
    return config_device


def export_embeddings(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    seed: int,
    device_override: str,
    config_path: Optional[Path],
    overrides: Optional[list[str]] = None,
) -> None:
    cfg = load_config(config_name="core", config_path=config_path, overrides=overrides or [])
    seed_everything(seed)

    device_choice = _device_choice(str(cfg.device), device_override)
    device_cfg = resolve_device(device_choice)
    device = device_cfg.device

    dataset_root = dataset_path
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    clip_cfg = cfg.training.clip
    data_cfg = cfg.data.synthetic

    transform = build_clip_image_transform(
        ClipTransformConfig(
            image_size=int(data_cfg.image.size),
            rotation=int(data_cfg.augment.rotation),
        )
    )

    loader = build_embedding_loader(
        root=dataset_root,
        batch_size=batch_size,
        num_workers=int(clip_cfg.num_workers),
        seed=seed,
        device_cfg=device_cfg,
        transform=transform,
    )

    model = ClipAlignmentModel(
        projection=ProjectionConfig(
            hidden_dim=int(clip_cfg.omics_encoder.hidden_dim),
            projection_dim=int(clip_cfg.projection_dim),
        ),
        omics_dim=int(data_cfg.omics.dim),
        image_projection=ProjectionConfig(
            hidden_dim=int(clip_cfg.image_encoder.hidden_dim),
            projection_dim=int(clip_cfg.projection_dim),
        ),
        temperature=float(clip_cfg.temperature),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    records: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            omics = batch["omics"].to(device)
            image_proj, omics_proj = model(images, omics)
            for idx, sample_id in enumerate(batch["sample_id"]):
                records.append(
                    {
                        "sample_id": sample_id,
                        "split": batch["split"][idx],
                        "label": int(batch["label"][idx].item()),
                        "image_embedding": image_proj[idx].cpu().tolist(),
                        "omics_embedding": omics_proj[idx].cpu().tolist(),
                    }
                )

    if not records:
        raise RuntimeError("No samples were exported; check dataset path")

    console.log("Exported embeddings", samples=len(records), output=str(output_path))

    df = pl.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    else:
        df.write_parquet(output_path)

    (output_path.with_suffix(".config.json")).write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
