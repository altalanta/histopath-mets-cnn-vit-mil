"""Image transforms used across pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torchvision import transforms


@dataclass
class SimCLRTransformConfig:
    image_size: int
    color_jitter: float
    blur_prob: float
    rotation: int


@dataclass
class ClipTransformConfig:
    image_size: int
    rotation: int


class SimCLRPairs:
    """Create two augmented views of the same image."""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, image):  # type: ignore[override]
        return self.base_transform(image), self.base_transform(image)


def build_simclr_transform(cfg: SimCLRTransformConfig) -> SimCLRPairs:
    color = transforms.ColorJitter(
        brightness=cfg.color_jitter,
        contrast=cfg.color_jitter,
        saturation=cfg.color_jitter,
        hue=min(0.2, cfg.color_jitter),
    )
    base = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=cfg.blur_prob),
            transforms.RandomRotation(cfg.rotation),
            transforms.ToTensor(),
        ]
    )
    return SimCLRPairs(base)


def build_clip_image_transform(cfg: ClipTransformConfig) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(int(cfg.image_size * 1.1)),
            transforms.CenterCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(cfg.rotation),
            transforms.ToTensor(),
        ]
    )
