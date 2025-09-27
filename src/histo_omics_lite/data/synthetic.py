"""Synthetic data generation for histo-omics-lite."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from pydantic import ValidationError
from rich.console import Console
from rich.progress import track

from .schemas import POLARS_SCHEMA, DatasetSummary, SampleRecord
from ..utils.config import load_config
from ..utils.device import resolve_device
from ..utils.seed import seed_everything

console = Console()

DEFAULT_LABELS = [
    "stroma",
    "tumor",
    "immune",
    "necrosis",
    "vasculature",
    "connective",
]


def _encode_png(array: np.ndarray) -> bytes:
    image = Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _encode_npy(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array.astype(np.float32), allow_pickle=False)
    buffer.seek(0)
    return buffer.read()


def _normalise_image(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor - tensor.min()
    denom = tensor.max() - tensor.min() + 1e-8
    return (tensor / denom).clamp(0.0, 1.0)


def _make_class_patterns(
    *,
    num_classes: int,
    channels: int,
    size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    xs = torch.linspace(-math.pi, math.pi, size, device=device)
    ys = torch.linspace(-math.pi, math.pi, size, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=0)
    patterns: list[torch.Tensor] = []
    for idx in range(num_classes):
        freq = idx + 1
        phase = idx * (math.pi / max(1, num_classes - 1))
        base = torch.sin(freq * grid[0] + phase) + torch.cos((freq + 1) * grid[1] - phase)
        color_weights = torch.linspace(0.7, 1.3, steps=channels, device=device).view(channels, 1, 1)
        pattern = base.unsqueeze(0) * color_weights
        pattern = F.avg_pool2d(pattern.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        patterns.append(pattern)
    stacked = torch.stack(patterns, dim=0)
    noise = torch.randn_like(stacked, generator=generator) * 0.05
    return stacked + noise


def _class_means(
    *,
    num_classes: int,
    dim: int,
    class_separation: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    base = torch.randn((num_classes, dim), generator=generator, device=device)
    base = torch.nn.functional.normalize(base, p=2, dim=1)
    return base * class_separation


def _resolve_label_names(num_classes: int) -> list[str]:
    if num_classes <= len(DEFAULT_LABELS):
        return DEFAULT_LABELS[:num_classes]
    padded = DEFAULT_LABELS + [f"class_{idx}" for idx in range(num_classes - len(DEFAULT_LABELS))]
    return padded[:num_classes]


def generate_synthetic_corpus(
    *,
    num_train: int,
    num_val: int,
    output_dir: Path,
    seed: int,
    device: str,
    config_path: Optional[Path],
    overrides: Optional[list[str]] = None,
) -> None:
    """Create turnkey synthetic dataset with validation guards."""

    cfg = load_config(config_name="core", config_path=config_path, overrides=overrides or [])
    data_cfg = cfg.data.synthetic

    train_samples = int(num_train or data_cfg.splits.train)
    val_samples = int(num_val or data_cfg.splits.val)

    if train_samples <= 0 or val_samples <= 0:
        raise ValueError("Sample counts must be positive")

    output_dir = output_dir.resolve()
    shards_dir = output_dir / "shards"
    tables_dir = output_dir / "tables"
    manifests_dir = output_dir / "manifests"
    shards_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device_cfg = resolve_device(device)
    generator = torch.Generator(device=device_cfg.device).manual_seed(seed)

    num_classes = int(data_cfg.omics.classes)
    if num_classes < 2:
        raise ValueError("At least two classes are required for contrastive training")

    label_names = _resolve_label_names(num_classes)
    image_size = int(data_cfg.image.size)
    channels = int(data_cfg.image.channels)
    transcript_dim = int(data_cfg.omics.dim)
    class_separation = float(data_cfg.omics.class_separation)
    omics_noise = float(data_cfg.omics.noise)
    samples_per_shard = int(data_cfg.shards.samples_per_shard)
    compression = data_cfg.shards.compression

    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")

    console.log(
        "Generating synthetic dataset",
        train=train_samples,
        val=val_samples,
        classes=num_classes,
        image_size=image_size,
        transcript_dim=transcript_dim,
    )

    patterns = _make_class_patterns(
        num_classes=num_classes,
        channels=channels,
        size=image_size,
        generator=generator,
        device=device_cfg.device,
    )
    transcript_means = _class_means(
        num_classes=num_classes,
        dim=transcript_dim,
        class_separation=class_separation,
        generator=generator,
        device=device_cfg.device,
    )

    shards_by_split = {
        "train": shards_dir / "train",
        "val": shards_dir / "val",
    }
    for path in shards_by_split.values():
        path.mkdir(parents=True, exist_ok=True)

    suffix = ".tar.gz" if compression == "gz" else ".tar"

    records: list[dict[str, object]] = []
    class_counts: dict[str, int] = defaultdict(int)

    split_to_count = {"train": train_samples, "val": val_samples}

    for split, count in split_to_count.items():
        shard_pattern = str(shards_by_split[split] / f"{split}-%06d{suffix}")
        with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as sink:
            for idx in track(range(count), description=f"{split} samples", console=console):
                label = int(torch.randint(num_classes, (1,), generator=generator).item())
                label_name = label_names[label]
                base_pattern = patterns[label]
                noise = torch.randn_like(base_pattern, generator=generator) * 0.2
                augmented = base_pattern + noise
                image_tensor = _normalise_image(augmented)
                image_np = (image_tensor * 255.0).clamp(0, 255).to(torch.uint8)
                image_bytes = _encode_png(image_np.permute(1, 2, 0).cpu().numpy())

                transcript = transcript_means[label] + torch.randn(
                    transcript_dim, generator=generator, device=device_cfg.device
                ) * omics_noise
                transcript_np = transcript.to(torch.float32).cpu().numpy()
                transcript_bytes = _encode_npy(transcript_np)

                sample_id = f"{split}_{idx:06d}"
                key = sample_id
                metadata = {
                    "sample_id": sample_id,
                    "split": split,
                    "label": label,
                    "label_name": label_name,
                }
                record = {
                    "__key__": key,
                    "png": image_bytes,
                    "npy": transcript_bytes,
                    "json": json.dumps(metadata).encode("utf-8"),
                }
                sink.write(record)

                shard_name = Path(shard_pattern % (idx // samples_per_shard)).name
                try:
                    validated = SampleRecord(
                        sample_id=sample_id,
                        split=split,
                        label=label,
                        label_name=label_name,
                        shard=Path(shard_name).name,
                        key=key,
                        image_mean=float(image_tensor.mean().item()),
                        omics_norm=float(torch.linalg.vector_norm(transcript).item()),
                    )
                except ValidationError as exc:
                    msg = f"Sample validation failed for {sample_id}: {exc}"
                    raise RuntimeError(msg) from exc

                records.append(validated.model_dump())
                class_counts[label_name] += 1

    try:
        df = pl.DataFrame(records)
        df = df.select(
            [
                pl.col("sample_id").cast(POLARS_SCHEMA["sample_id"]),
                pl.col("split").cast(POLARS_SCHEMA["split"]),
                pl.col("label").cast(POLARS_SCHEMA["label"]),
                pl.col("label_name").cast(POLARS_SCHEMA["label_name"]),
                pl.col("shard").cast(POLARS_SCHEMA["shard"]),
                pl.col("key").cast(POLARS_SCHEMA["key"]),
                pl.col("image_mean").cast(POLARS_SCHEMA["image_mean"]),
                pl.col("omics_norm").cast(POLARS_SCHEMA["omics_norm"]),
            ]
        )
    except pl.exceptions.PolarsError as exc:
        msg = "Failed to coerce records into the expected polars schema"
        raise RuntimeError(msg) from exc

    df.write_parquet(tables_dir / "samples.parquet")

    summary = DatasetSummary(
        total_samples=train_samples + val_samples,
        train_samples=train_samples,
        val_samples=val_samples,
        num_classes=num_classes,
        image_size=image_size,
        transcript_dim=transcript_dim,
    )

    manifest = {
        "summary": summary.model_dump(),
        "class_distribution": dict(class_counts),
        "config": {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "image_size": image_size,
            "transcript_dim": transcript_dim,
            "num_classes": num_classes,
            "samples_per_shard": samples_per_shard,
            "compression": compression,
        },
    }

    (manifests_dir / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    console.log(f"Synthetic dataset ready at {output_dir}")
