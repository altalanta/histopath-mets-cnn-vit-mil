"""Retrieval evaluation pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
from rich.console import Console

from ..data.pipeline import load_split
from ..data.transforms import ClipTransformConfig, build_clip_image_transform
from ..utils.config import load_config, to_dict
from ..utils.device import resolve_device
from ..utils.seed import seed_everything
from .gradcam import GradCAM
from ..training.models import ClipAlignmentModel, ProjectionConfig

console = Console()


def _device_choice(config_device: str, override: str) -> str:
    if override != "auto":
        return override
    return config_device


def _topk(similarity: torch.Tensor, k: int) -> float:
    k = min(k, similarity.size(1))
    targets = torch.arange(similarity.size(0), device=similarity.device)
    topk = similarity.topk(k, dim=1).indices
    hits = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


def _bootstrap_ci(
    *,
    similarity: torch.Tensor,
    k: int,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    sim_np = similarity.cpu().numpy()
    n = sim_np.shape[0]
    for _ in range(samples):
        indices = rng.choice(n, size=n, replace=True)
        subset = sim_np[np.ix_(indices, indices)]
        subset_tensor = torch.from_numpy(subset)
        scores.append(_topk(subset_tensor, k))
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(scores, alpha / 2))
    upper = float(np.quantile(scores, 1 - alpha / 2))
    return lower, upper


def _build_model(cfg, data_cfg, clip_cfg, device: torch.device) -> ClipAlignmentModel:
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
    )
    return model.to(device)


def _save_umap(
    *,
    image_embeddings: torch.Tensor,
    omics_embeddings: torch.Tensor,
    labels: list[str],
    output: Path,
    umap_cfg,
    seed: int,
) -> None:
    reducer = umap.UMAP(
        n_neighbors=int(umap_cfg.n_neighbors),
        min_dist=float(umap_cfg.min_dist),
        metric=str(umap_cfg.metric),
        random_state=seed,
    )
    combined = torch.cat([image_embeddings, omics_embeddings], dim=0).cpu().numpy()
    embedding = reducer.fit_transform(combined)
    num_samples = image_embeddings.size(0)
    modalities = ["image"] * num_samples + ["omics"] * num_samples

    palette = sns.color_palette("tab10", n_colors=len(set(labels)))
    label_to_color = {label: palette[idx % len(palette)] for idx, label in enumerate(sorted(set(labels)))}

    plt.figure(figsize=(6, 5))
    for modality in {"image", "omics"}:
        modality_offset = 0 if modality == "image" else num_samples
        xs = []
        ys = []
        colors = []
        for idx in range(num_samples):
            point = embedding[modality_offset + idx]
            xs.append(point[0])
            ys.append(point[1])
            colors.append(label_to_color[labels[idx]])
        marker = "o" if modality == "image" else "^"
        plt.scatter(xs, ys, c=colors, marker=marker, alpha=0.7, label=modality)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=label) for label, color in label_to_color.items()]
    plt.legend(handles=handles, title="Tissue", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("UMAP of image/omics embeddings")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def _save_gradcam_overlays(
    *,
    model: ClipAlignmentModel,
    samples,
    transformed,
    device: torch.device,
    output_dir: Path,
    top_k: int,
) -> list[str]:
    target_layer = model.image_model.encoder.feature_extractor[-2]
    gradcam = GradCAM(model, target_layer)
    saved: list[str] = []
    for sample, transformed_tensor in zip(samples[:top_k], transformed[:top_k]):

        def image_fn() -> torch.Tensor:
            return transformed_tensor.unsqueeze(0).to(device)

        def score_fn(image_tensor: torch.Tensor) -> torch.Tensor:
            omics_tensor = sample.omics.unsqueeze(0).to(device)
            image_proj, omics_proj = model(image_tensor, omics_tensor)
            return torch.sum(image_proj * omics_proj)

        result = gradcam.generate(image_fn=image_fn, score_fn=score_fn)
        heatmap_tensor = torch.from_numpy(result.heatmap).unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(
            heatmap_tensor,
            size=transformed_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        fig = plt.figure(figsize=(3, 3))
        plt.imshow(sample.image)
        plt.imshow(heatmap_resized, cmap="inferno", alpha=0.4)
        plt.title(f"{sample.sample_id} | score={result.score:.2f}")
        plt.axis("off")
        plt.tight_layout()
        output_path = output_dir / f"gradcam_{sample.sample_id}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        saved.append(str(output_path))
    gradcam.close()
    return saved


def run_retrieval_eval(
    *,
    checkpoint_path: Path,
    data_root: Optional[Path],
    config_path: Optional[Path],
    seed: int,
    output_dir: Path,
    device_override: str,
    overrides: Optional[list[str]] = None,
) -> None:
    cfg = load_config(config_name="core", config_path=config_path, overrides=overrides or [])
    seed_everything(seed)

    device_choice = _device_choice(str(cfg.device), device_override)
    device_cfg = resolve_device(device_choice)
    device = device_cfg.device

    data_cfg = cfg.data.synthetic
    clip_cfg = cfg.training.clip
    eval_cfg = cfg.evaluation.retrieval

    dataset_root = data_root or Path(cfg.paths.synthetic_root)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(cfg, data_cfg, clip_cfg, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    val_samples = load_split(dataset_root, "val")
    if not val_samples:
        raise RuntimeError("Validation split is empty; cannot run retrieval evaluation")

    transform = build_clip_image_transform(
        ClipTransformConfig(
            image_size=int(data_cfg.image.size),
            rotation=int(data_cfg.augment.rotation),
        )
    )

    image_embeddings = []
    omics_embeddings = []
    labels = []
    transformed_images: list[torch.Tensor] = []

    for sample in val_samples:
        transformed = transform(sample.image)
        transformed_images.append(transformed)
        image_tensor = transformed.unsqueeze(0).to(device)
        omics_tensor = sample.omics.unsqueeze(0).to(device)
        with torch.no_grad():
            image_proj, omics_proj = model(image_tensor, omics_tensor)
        image_embeddings.append(image_proj.cpu())
        omics_embeddings.append(omics_proj.cpu())
        labels.append(sample.label_name)

    image_embeddings_tensor = torch.cat(image_embeddings)
    omics_embeddings_tensor = torch.cat(omics_embeddings)
    similarity = image_embeddings_tensor @ omics_embeddings_tensor.t()

    metrics = {
        "top1": _topk(similarity, 1),
        "top5": _topk(similarity, 5),
    }

    ci = {
        "top1": _bootstrap_ci(
            similarity=similarity,
            k=1,
            samples=int(eval_cfg.bootstrap_samples),
            confidence_level=float(eval_cfg.confidence_level),
            seed=seed,
        ),
        "top5": _bootstrap_ci(
            similarity=similarity,
            k=5,
            samples=int(eval_cfg.bootstrap_samples),
            confidence_level=float(eval_cfg.confidence_level),
            seed=seed + 1,
        ),
    }

    report = {
        "metrics": metrics,
        "confidence_intervals": {key: {"lower": lower, "upper": upper} for key, (lower, upper) in ci.items()},
        "config": to_dict(cfg),
        "num_samples": len(val_samples),
        "checkpoint": str(checkpoint_path),
    }

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _save_umap(
        image_embeddings=image_embeddings_tensor,
        omics_embeddings=omics_embeddings_tensor,
        labels=labels,
        output=output_dir / "umap.png",
        umap_cfg=eval_cfg.umap,
        seed=seed,
    )

    gradcam_paths = _save_gradcam_overlays(
        model=model,
        samples=val_samples,
        transformed=transformed_images,
        device=device,
        output_dir=output_dir,
        top_k=int(eval_cfg.gradcam.top_k),
    )

    embeddings_records = [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "label": sample.label_name,
            "image_embedding": image_embeddings_tensor[idx].tolist(),
            "omics_embedding": omics_embeddings_tensor[idx].tolist(),
        }
        for idx, sample in enumerate(val_samples)
    ]
    (output_dir / "embeddings.json").write_text(json.dumps(embeddings_records, indent=2), encoding="utf-8")

    console.log(
        "Retrieval evaluation complete",
        top1=f"{metrics['top1']:.3f}",
        top5=f"{metrics['top5']:.3f}",
        overlays=len(gradcam_paths),
    )
