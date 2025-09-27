"""Data loading utilities built on WebDataset."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from rich.console import Console
from torch.utils.data import DataLoader, Dataset

from .transforms import ClipTransformConfig, SimCLRTransformConfig, build_clip_image_transform, build_simclr_transform
from ..utils.device import DeviceConfig
from ..utils.seed import make_worker_init_fn

console = Console()


@dataclass(frozen=True)
class SyntheticSample:
    split: str
    sample_id: str
    label: int
    label_name: str
    image: Image.Image
    omics: torch.Tensor


def _decode_json(payload: object) -> dict[str, object]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unsupported JSON payload type: {type(payload)}")


def _decode_npy(payload: object) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(payload).float()
    if isinstance(payload, bytes):
        buffer = BytesIO(payload)
        array = np.load(buffer)
        return torch.from_numpy(array).float()
    raise TypeError(f"Unsupported npy payload type: {type(payload)}")


def _decode_image(payload: object) -> Image.Image:
    if isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, bytes):
        return Image.open(BytesIO(payload)).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(payload)}")


def load_split(root: Path, split: str) -> list[SyntheticSample]:
    pattern = str(root / "shards" / split / "*.tar*")
    dataset = wds.WebDataset(pattern, shardshuffle=False, handler=wds.handlers.reraise).decode("pil")

    start = time.perf_counter()
    samples: list[SyntheticSample] = []
    for sample in dataset:  # type: ignore[assignment]
        image = _decode_image(sample["png"])
        omics = _decode_npy(sample["npy"])
        meta = _decode_json(sample["json"])
        samples.append(
            SyntheticSample(
                split=split,
                sample_id=str(meta["sample_id"]),
                label=int(meta["label"]),
                label_name=str(meta["label_name"]),
                image=image,
                omics=omics,
            )
        )
    duration = time.perf_counter() - start
    throughput = len(samples) / duration if duration > 0 else float("inf")
    console.log(f"Loaded {len(samples)} {split} samples", throughput=f"{throughput:0.1f}/s", pattern=pattern)
    return samples


class SimCLRDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: Sequence[SyntheticSample], transform_cfg: SimCLRTransformConfig):
        self.samples = list(samples)
        self.transform = build_simclr_transform(transform_cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        view1, view2 = self.transform(sample.image)
        return {
            "view1": view1,
            "view2": view2,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
        }


class ClipDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: Sequence[SyntheticSample], transform_cfg: ClipTransformConfig):
        self.samples = list(samples)
        self.transform = build_clip_image_transform(transform_cfg)

    def __len__(self) -> int:\n        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self.transform(sample.image)
        return {
            "image": image,
            "omics": sample.omics,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
        }


class EmbeddingDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, samples: Sequence[SyntheticSample], transform: Callable[[Image.Image], torch.Tensor]):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self.transform(sample.image)
        return {
            "image": image,
            "omics": sample.omics,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_id,
            "split": sample.split,
        }


def build_simclr_dataloader(
    *,
    root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device_cfg: DeviceConfig,
    transform_cfg: SimCLRTransformConfig,
) -> DataLoader[dict[str, torch.Tensor | str]]:
    samples = load_split(root, "train")
    dataset = SimCLRDataset(samples, transform_cfg)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=device_cfg.use_cuda,
        generator=generator,
    )


def build_clip_dataloaders(
    *,
    root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device_cfg: DeviceConfig,
    transform_cfg: ClipTransformConfig,
) -> tuple[DataLoader[dict[str, torch.Tensor | str]], DataLoader[dict[str, torch.Tensor | str]]]:
    train_samples = load_split(root, "train")
    val_samples = load_split(root, "val")

    train_dataset = ClipDataset(train_samples, transform_cfg)
    val_dataset = ClipDataset(val_samples, transform_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=device_cfg.use_cuda,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=make_worker_init_fn(seed + 1),
        pin_memory=device_cfg.use_cuda,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return train_loader, val_loader


def build_embedding_loader(
    *,
    root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    device_cfg: DeviceConfig,
    transform: Callable[[Image.Image], torch.Tensor],
) -> DataLoader[dict[str, torch.Tensor | str]]:
    all_samples = load_split(root, "train") + load_split(root, "val")
    dataset = EmbeddingDataset(all_samples, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=make_worker_init_fn(seed),
        pin_memory=device_cfg.use_cuda,
        generator=torch.Generator().manual_seed(seed),
    )
