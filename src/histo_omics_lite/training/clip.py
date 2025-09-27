"""CLIP-style alignment training."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console

from ..data.pipeline import build_clip_dataloaders
from ..data.transforms import ClipTransformConfig
from ..utils.config import load_config, to_dict
from ..utils.device import resolve_device
from ..utils.profiler import maybe_profile
from ..utils.seed import seed_everything
from .models import ClipAlignmentModel, ProjectionConfig, identical_temperature_loss

console = Console()


def _device_choice(config_device: str, override: str) -> str:
    if override != "auto":
        return override
    return config_device


def _load_simclr_weights(model: ClipAlignmentModel, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("model_state", payload)
    missing, unexpected = model.image_model.load_state_dict(state_dict, strict=False)
    if missing:
        console.log("Missing keys while loading SimCLR weights", missing=missing)
    if unexpected:
        console.log("Unexpected keys while loading SimCLR weights", unexpected=unexpected)


def run_clip(
    *,
    config_path: Optional[Path],
    seed: int,
    output_dir: Path,
    device_override: str,
    simclr_checkpoint: Optional[Path],
    overrides: Optional[list[str]] = None,
) -> None:
    cfg = load_config(config_name="core", config_path=config_path, overrides=overrides or [])
    seed_everything(seed)

    device_choice = _device_choice(str(cfg.device), device_override)
    device_cfg = resolve_device(device_choice)
    device = device_cfg.device

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_cfg = cfg.training.clip
    data_cfg = cfg.data.synthetic

    transform_cfg = ClipTransformConfig(
        image_size=int(data_cfg.image.size),
        rotation=int(data_cfg.augment.rotation),
    )

    batch_size = int(clip_cfg.batch_size)

    train_loader, val_loader = build_clip_dataloaders(
        root=Path(cfg.paths.synthetic_root),
        batch_size=batch_size,
        num_workers=int(clip_cfg.num_workers),
        seed=seed,
        device_cfg=device_cfg,
        transform_cfg=transform_cfg,
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

    if simclr_checkpoint:
        _load_simclr_weights(model, simclr_checkpoint)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(clip_cfg.base_learning_rate),
        weight_decay=float(clip_cfg.weight_decay),
    )

    log_interval = int(clip_cfg.log_interval)
    epochs = int(clip_cfg.epochs)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_path = output_dir / "clip_best.pt"

    profiler_path = output_dir / "profiler_trace.json"

    console.log("Starting CLIP alignment", epochs=epochs, device=str(device))

    with maybe_profile(
        enabled=bool(cfg.profiling.enabled),
        activities=list(cfg.profiling.activities),
        output=profiler_path,
    ) as profiler:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batch_start = time.perf_counter()
            for batch_idx, batch in enumerate(train_loader, start=1):
                images = batch["image"].to(device)
                omics = batch["omics"].to(device)

                img_proj, omics_proj = model(images, omics)
                loss = identical_temperature_loss(img_proj, omics_proj, model.logit_scale)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                model.logit_scale.data.clamp_(math.log(1 / 100.0), math.log(100.0))

                epoch_loss += float(loss.item())

                if profiler is not None:
                    profiler.step()

                if batch_idx % log_interval == 0:
                    elapsed = time.perf_counter() - batch_start
                    samples = log_interval * batch_size
                    throughput = samples / elapsed if elapsed > 0 else 0.0
                    console.log(
                        f"epoch={epoch + 1} step={batch_idx}",
                        loss=f"{loss.item():.4f}",
                        throughput=f"{throughput:0.1f} pairs/s",
                    )
                    batch_start = time.perf_counter()

            avg_train_loss = epoch_loss / max(1, len(train_loader))

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                all_image = []
                all_omics = []
                for batch in val_loader:
                    images = batch["image"].to(device)
                    omics = batch["omics"].to(device)
                    img_proj, omics_proj = model(images, omics)
                    val_loss += float(identical_temperature_loss(img_proj, omics_proj, model.logit_scale).item())
                    all_image.append(img_proj.cpu())
                    all_omics.append(omics_proj.cpu())
                val_loss = val_loss / max(1, len(val_loader))

                if all_image:
                    image_stack = torch.cat(all_image)
                    omics_stack = torch.cat(all_omics)
                    similarity = image_stack @ omics_stack.t()
                    preds = similarity.argmax(dim=1)
                    labels = torch.arange(similarity.size(0))
                    val_top1 = float((preds == labels).float().mean().item())
                else:
                    val_top1 = 0.0

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_top1": val_top1,
                }
            )

            console.log(
                f"epoch {epoch + 1} | train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} top1={val_top1:.3f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": to_dict(cfg),
                    },
                    best_path,
                )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
    console.log("CLIP alignment complete", best_checkpoint=str(best_path), loss=f"{best_loss:.4f}")
