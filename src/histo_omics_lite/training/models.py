"""Model components used for SimCLR and CLIP training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass(frozen=True)
class ProjectionConfig:
    hidden_dim: int
    projection_dim: int


class ResNetEncoder(nn.Module):
    """ResNet-18 backbone returning penultimate features."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        modules = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        self.out_dim = backbone.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.feature_extractor(x)
        return torch.flatten(feats, 1)


class ProjectionHead(nn.Module):
    def __init__(self, cfg: ProjectionConfig, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.BatchNorm1d(cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(x)
        return F.normalize(out, dim=1)


class SimCLRModel(nn.Module):
    def __init__(self, projection: ProjectionConfig) -> None:
        super().__init__()
        self.encoder = ResNetEncoder()
        self.projection = ProjectionHead(projection, self.encoder.out_dim)

    @property
    def feature_dim(self) -> int:
        return self.encoder.out_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


class OmicsEncoder(nn.Module):
    def __init__(self, input_dim: int, projection: ProjectionConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection.hidden_dim),
            nn.LayerNorm(projection.hidden_dim),
            nn.GELU(),
            nn.Linear(projection.hidden_dim, projection.projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.normalize(self.net(x), dim=1)


class ClipAlignmentModel(nn.Module):
    def __init__(
        self,
        *,
        projection: ProjectionConfig,
        omics_dim: int,
        image_projection: ProjectionConfig,
        temperature: float,
    ) -> None:
        super().__init__()
        self.image_model = SimCLRModel(image_projection)
        self.omics_model = OmicsEncoder(omics_dim, projection)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, image: torch.Tensor, omics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, image_proj = self.image_model(image)
        omics_proj = self.omics_model(omics)
        return image_proj, omics_proj

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        _, image_proj = self.image_model(image)
        return image_proj

    def encode_omics(self, omics: torch.Tensor) -> torch.Tensor:
        return self.omics_model(omics)


def identical_temperature_loss(image_embeddings: torch.Tensor, omics_embeddings: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """CLIP-style symmetric cross-entropy loss."""

    logits = logit_scale.exp() * image_embeddings @ omics_embeddings.t()
    labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2.0


def simclr_nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard NT-Xent loss used by SimCLR."""

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    logits = similarity / temperature
    mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -9e15)
    labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)], dim=0)
    labels = labels.to(z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss
